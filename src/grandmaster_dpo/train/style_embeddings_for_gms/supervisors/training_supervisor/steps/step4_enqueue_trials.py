from __future__ import annotations

import math
import time
from typing import Any, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.llm_policy import build_common_step_context, call_step_llm, record_step_decision
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.plan_view import compute_training_coverage
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ProposalSpec, QueueItem, TrainingSupervisorState, evolve_state


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()
    max_adds = _max_adds(state)

    candidates = _build_enqueue_candidates(state, created_at=current_time)
    if not candidates:
        return evolve_state(
            state,
            last_step="step4_enqueue_trials",
            last_action="enqueue_trials:no_candidates",
        )

    common = build_common_step_context(state)
    step_context = {
        **common,
        "candidate_queue_additions": candidates,
        "max_queue_additions": max_adds,
    }

    llm_result = call_step_llm(
        state=state,
        step_name="step4_enqueue_trials",
        system_goal="Choose which already-defined studies or proposed studies should actually be queued now for training.",
        allowed_actions=["enqueue_none", "enqueue_trials", "wait"],
        decision_schema_hint={
            "action": "enqueue_trials",
            "reason": "string",
            "reasoning_summary": "string",
            "candidate_ids": ["string"],
            "priority_overrides": {"candidate_id": 5},
            "reason_overrides": {"candidate_id": "string"},
            "shared_context_updates": {},
        },
        step_context=step_context,
    )

    decision = dict(llm_result["decision"])
    scratch = record_step_decision(
        state,
        step_name="step4_enqueue_trials",
        llm_result=llm_result,
        step_context_snapshot=step_context,
    )

    if decision.get("action") != "enqueue_trials":
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step4_enqueue_trials",
            last_action=f"enqueue_trials:{decision.get('action', 'enqueue_none')}",
        )

    valid_ids = {c["candidate_id"] for c in candidates}
    raw_ids = list(decision.get("candidate_ids", []))
    str_ids = [str(x) for x in raw_ids]
    if not str_ids:
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step4_enqueue_trials",
            last_action="enqueue_trials:empty_candidate_ids",
        )
    if any(x not in valid_ids for x in str_ids):
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step4_enqueue_trials",
            last_action="enqueue_trials:invalid_candidate_ids",
        )

    selected_ids = str_ids[:max_adds]
    priority_overrides = decision.get("priority_overrides", {})
    reason_overrides = decision.get("reason_overrides", {})

    queue = list(state.queue)
    existing_identities = {item.identity() for item in queue}

    added = 0
    for candidate_id in selected_ids:
        candidate = next((c for c in candidates if c["candidate_id"] == candidate_id), None)
        if candidate is None:
            continue

        item = _candidate_to_queue_item(
            candidate=candidate,
            default_reason=str(decision.get("reason", "llm_enqueue")),
            priority_overrides=priority_overrides if isinstance(priority_overrides, Mapping) else {},
            reason_overrides=reason_overrides if isinstance(reason_overrides, Mapping) else {},
        )
        if item.identity() in existing_identities:
            continue
        queue.append(item)
        existing_identities.add(item.identity())
        added += 1

    queue.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
    return evolve_state(
        state,
        queue=tuple(queue),
        scratch=scratch,
        last_step="step4_enqueue_trials",
        last_action=f"enqueue_trials:added={added}",
    )


def _build_enqueue_candidates(
    state: TrainingSupervisorState,
    *,
    created_at: float,
) -> list[dict[str, Any]]:
    coverage = compute_training_coverage(
        state.plan_view,
        state.registry_summary,
        queue=state.queue,
        active_run=state.active_run,
    )

    queue_identities = {item.identity() for item in state.queue}
    terminal_studies = {
        record.study_name
        for record in (
            list(state.registry_summary.completed_runs) + list(state.registry_summary.failed_or_killed_runs)
        )
    }
    if state.active_run is not None:
        terminal_studies.add(state.active_run.study_name)

    candidates: list[dict[str, Any]] = []

    for proposal in state.proposed_studies:
        identity = proposal.new_study_name
        if identity in queue_identities or identity in terminal_studies:
            continue
        candidates.append(
            {
                "candidate_id": f"proposal:{proposal.new_study_name}",
                "source": "proposal",
                "default_priority": 7,
                "default_reason": "proposed_training_trial",
                "proposal": proposal.to_dict(),
                "created_at": created_at,
            }
        )

    for axis, status in coverage.items():
        for value in status.remaining:
            for study_name in _defined_studies_for_gap(axis=axis, value=value):
                if study_name in queue_identities or study_name in terminal_studies:
                    continue
                candidates.append(
                    {
                        "candidate_id": f"study:{study_name}",
                        "source": "predefined",
                        "default_priority": _priority_for_axis(axis),
                        "default_reason": f"fills_required_{axis}",
                        "study_name": study_name,
                        "axis": axis,
                        "axis_value": value,
                        "created_at": created_at,
                    }
                )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["candidate_id"] in seen:
            continue
        seen.add(candidate["candidate_id"])
        deduped.append(candidate)

    deduped.sort(key=lambda row: (-int(row.get("default_priority", 0)), row["candidate_id"]))
    return deduped


def _candidate_to_queue_item(
    *,
    candidate: Mapping[str, Any],
    default_reason: str,
    priority_overrides: Mapping[str, Any],
    reason_overrides: Mapping[str, Any],
) -> QueueItem:
    candidate_id = str(candidate["candidate_id"])
    priority = priority_overrides.get(candidate_id, candidate.get("default_priority", 0))
    reason = str(reason_overrides.get(candidate_id, candidate.get("default_reason", default_reason)))
    created_at = float(candidate.get("created_at", time.time()))
    source = str(candidate.get("source", "llm_enqueue"))

    if candidate.get("source") == "proposal":
        proposal = ProposalSpec.from_dict(candidate["proposal"])
        if proposal.new_study_name in STUDIES:
            return QueueItem(
                study_name=proposal.new_study_name,
                priority=int(priority),
                created_at=created_at,
                source=source,
                reason=reason,
            )
        return QueueItem(
            proposal=proposal,
            priority=int(priority),
            created_at=created_at,
            source=source,
            reason=reason,
        )

    return QueueItem(
        study_name=str(candidate["study_name"]),
        priority=int(priority),
        created_at=created_at,
        source=source,
        reason=reason,
    )


def _defined_studies_for_gap(*, axis: str, value: Any) -> list[str]:
    out: list[str] = []
    for study_name, cfg in STUDIES.items():
        meta = _cfg_to_meta(study_name, cfg.to_dict())
        if _study_fills_gap(axis=axis, value=value, meta=meta):
            out.append(study_name)
    return out


def _study_fills_gap(*, axis: str, value: Any, meta: Mapping[str, Any]) -> bool:
    if axis == "tau":
        return (
            _infer_stage(str(meta.get("study_name", ""))) == "screen"
            and meta.get("pair_variant") == "v1"
            and meta.get("phi_variant") == "phi0"
            and _same_number(meta.get("tau"), value)
        )
    if axis == "phi_variant":
        return (
            _infer_stage(str(meta.get("study_name", ""))) == "screen"
            and meta.get("pair_variant") == "v1"
            and meta.get("phi_variant") == value
        )
    if axis == "batch_size":
        return _infer_stage(str(meta.get("study_name", ""))) == "ablation" and _same_number(
            meta.get("batch_size"), value
        )
    if axis == "lr":
        return _infer_stage(str(meta.get("study_name", ""))) == "ablation" and _same_number(
            meta.get("lr"), value
        )
    if axis == "pair_variant":
        return meta.get("pair_variant") == value
    return False


def _max_adds(state: TrainingSupervisorState) -> int:
    value = state.plan_view.scheduler_policy.get("max_new_queue_items_per_decision", 2)
    try:
        return max(1, int(value))
    except Exception:
        return 2


def _priority_for_axis(axis: str) -> int:
    if axis == "tau":
        return 9
    if axis == "phi_variant":
        return 8
    if axis in {"batch_size", "lr"}:
        return 6
    if axis == "pair_variant":
        return 5
    return 4


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _infer_stage(study_name: str) -> str | None:
    if study_name.startswith("debug_"):
        return "debug"
    if study_name.startswith("screen_"):
        return "screen"
    if study_name.startswith("final_") or study_name.startswith("finalist_"):
        return "finalist"
    if study_name.startswith("ablation_") or study_name.startswith("ablate_"):
        return "ablation"
    return None


def _cfg_to_meta(study_name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    return {
        "study_name": study_name,
        "pair_variant": cfg.get("pair_variant"),
        "tau": cfg.get("tau"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "max_steps_per_epoch": cfg.get("max_steps_per_epoch"),
        "max_eval_batches": cfg.get("max_eval_batches"),
        "phi_variant": model.get("variant_name"),
    }