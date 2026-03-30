from __future__ import annotations

import time
from typing import Any, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.llm_policy import build_common_step_context, call_step_llm, record_step_decision
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import append_proposed_study
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.plan_view import compute_training_coverage
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ProposalSpec, TrainingSupervisorState, evolve_state


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()
    max_new = _max_new_proposals(state)

    coverage = compute_training_coverage(
        state.plan_view,
        state.registry_summary,
        queue=state.queue,
        active_run=state.active_run,
    )

    common = build_common_step_context(state)
    step_context = {
        **common,
        "coverage": {axis: status.to_dict() for axis, status in coverage.items()},
        "candidate_base_studies": [
            record.study_name for record in state.registry_summary.top_completed_runs
        ] + ([state.active_run.study_name] if state.active_run else []),
        "existing_defined_studies": sorted(STUDIES.keys())[:500],
        "proposal_count_limit": max_new,
    }

    llm_result = call_step_llm(
        state=state,
        step_name="step3_propose_trials",
        system_goal="Propose only training studies that improve missing ablation coverage or worthwhile local follow-up around the strongest current training recipe.",
        allowed_actions=["no_new_proposals", "propose_trials", "wait"],
        decision_schema_hint={
            "action": "propose_trials",
            "reason": "string",
            "reasoning_summary": "string",
            "proposals": [
                {
                    "base_study": "string",
                    "new_study_name": "string",
                    "changes": {},
                }
            ],
            "shared_context_updates": {},
        },
        step_context=step_context,
    )

    decision = dict(llm_result["decision"])
    scratch = record_step_decision(
        state,
        step_name="step3_propose_trials",
        llm_result=llm_result,
        step_context_snapshot=step_context,
    )

    if decision.get("action") != "propose_trials":
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step3_propose_trials",
            last_action=f"propose_trials:{decision.get('action', 'no_new_proposals')}",
        )

    existing_names = {proposal.new_study_name for proposal in state.proposed_studies}
    validated: list[ProposalSpec] = []

    for raw in list(decision.get("proposals", []))[:max_new]:
        if not isinstance(raw, Mapping):
            continue
        try:
            proposal = ProposalSpec.from_dict(raw)
        except Exception:
            continue
        if not _proposal_is_valid(proposal, state):
            continue
        if proposal.new_study_name in existing_names:
            continue
        existing_names.add(proposal.new_study_name)
        validated.append(proposal)
        append_proposed_study(
            proposal,
            reason=str(decision.get("reason", "llm_proposal")),
            source="training_supervisor_llm",
            created_at=current_time,
        )

    combined = list(state.proposed_studies)
    combined.extend(validated)

    return evolve_state(
        state,
        proposed_studies=tuple(combined),
        scratch=scratch,
        last_step="step3_propose_trials",
        last_action=f"propose_trials:added={len(validated)}",
    )


def _max_new_proposals(state: TrainingSupervisorState) -> int:
    value = state.plan_view.scheduler_policy.get("max_new_queue_items_per_decision", 2)
    try:
        return max(1, int(value))
    except Exception:
        return 2


def _proposal_is_valid(proposal: ProposalSpec, state: TrainingSupervisorState) -> bool:
    if proposal.base_study not in STUDIES:
        return False
    if proposal.new_study_name in STUDIES:
        return False

    base_cfg = STUDIES[proposal.base_study]
    changes = proposal.changes

    pair_variant = changes.get("pair_variant", getattr(base_cfg, "pair_variant", None))
    phi_variant = changes.get("phi_variant", getattr(getattr(base_cfg, "model", None), "variant_name", None))
    tau = changes.get("tau", getattr(base_cfg, "tau", None))
    batch_size = changes.get("batch_size", getattr(base_cfg, "batch_size", None))
    lr = changes.get("lr", getattr(base_cfg, "lr", None))

    if pair_variant is not None and _value_blocked("pair_variant", pair_variant, state):
        return False
    if phi_variant is not None and _value_blocked("phi_variant", phi_variant, state):
        return False

    allowed_axes = state.plan_view.allowed_axes
    if tau is not None and "tau" in allowed_axes and tau not in allowed_axes["tau"]:
        return False
    if batch_size is not None and "batch_size" in allowed_axes and batch_size not in allowed_axes["batch_size"]:
        return False
    if lr is not None and "lr" in allowed_axes and lr not in allowed_axes["lr"]:
        return False
    if phi_variant is not None and "phi_variant" in allowed_axes and phi_variant not in allowed_axes["phi_variant"]:
        return False
    if pair_variant is not None and "pair_variant" in allowed_axes and pair_variant not in allowed_axes["pair_variant"]:
        return False

    return True


def _value_blocked(axis: str, value: Any, state: TrainingSupervisorState) -> bool:
    blocked_axes = state.plan_view.blocked_axes
    currently_available = state.plan_view.currently_available

    if axis == "phi_variant":
        blocked = blocked_axes.get(str(value), {})
        if isinstance(blocked, Mapping) and blocked.get("blocked") is True:
            return True
        allowed = currently_available.get("phi_variants")
        if isinstance(allowed, list) and value not in allowed:
            return True

    if axis == "pair_variant":
        blocked = blocked_axes.get(str(value), {})
        if isinstance(blocked, Mapping) and blocked.get("blocked") is True:
            return True
        allowed = currently_available.get("pair_variants")
        if isinstance(allowed, list) and value not in allowed:
            return True

    return False
