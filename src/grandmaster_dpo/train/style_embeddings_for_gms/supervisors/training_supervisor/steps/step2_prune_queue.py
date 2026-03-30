from __future__ import annotations

import time
from collections import Counter
from typing import Any, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.llm_policy import build_common_step_context, call_step_llm, record_step_decision
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import QueueItem, RegistryEvent, TrainingSupervisorState, evolve_state


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()

    if not state.queue:
        return evolve_state(
            state,
            last_step="step2_prune_queue",
            last_action="prune_queue:empty",
        )

    # Deterministic pruning: if an item is already terminal according to
    # `run_registry.jsonl`, it should not remain queued even if the LLM
    # is unavailable.
    terminal_studies = {
        record.study_name
        for record in (
            list(state.registry_summary.completed_runs) + list(state.registry_summary.failed_or_killed_runs)
        )
        if record.study_name
    }

    if terminal_studies:
        kept: list[QueueItem] = []
        dropped: list[QueueItem] = []

        for item in state.queue:
            study_name = item.study_name or (item.proposal.new_study_name if item.proposal else None)
            if study_name and study_name in terminal_studies:
                dropped.append(item)
            else:
                kept.append(item)

        if dropped:
            for item in dropped:
                append_registry_event(
                    RegistryEvent(
                        time=current_time,
                        event="queue_item_pruned",
                        study_name=item.study_name or (item.proposal.new_study_name if item.proposal else None),
                        reason="already_terminal",
                        payload={
                            "queue_source": item.source,
                            "queue_reason": item.reason,
                            "queue_identity": item.identity(),
                        },
                    )
                )

            kept.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
            return evolve_state(
                state,
                queue=tuple(kept),
                last_step="step2_prune_queue",
                last_action=f"prune_queue:deterministic_dropped_terminal={len(dropped)}",
            )

    review_rows = _build_queue_review_rows(state)
    common = build_common_step_context(state)
    step_context = {
        **common,
        "queue_review": review_rows,
        "terminal_studies": [
            record.study_name
            for record in (
                list(state.registry_summary.completed_runs) + list(state.registry_summary.failed_or_killed_runs)
            )
        ],
    }

    llm_result = call_step_llm(
        state=state,
        step_name="step2_prune_queue",
        system_goal="Remove queued studies that are redundant, invalid, blocked, already terminal, or clearly lower-value than the rest of the current queue.",
        allowed_actions=["keep_queue", "drop_items", "wait"],
        decision_schema_hint={
            "action": "drop_items",
            "reason": "string",
            "reasoning_summary": "string",
            "drop_identities": ["string"],
            "shared_context_updates": {},
        },
        step_context=step_context,
    )

    decision = dict(llm_result["decision"])
    scratch = record_step_decision(
        state,
        step_name="step2_prune_queue",
        llm_result=llm_result,
        step_context_snapshot=step_context,
    )

    if decision.get("action") not in {"drop_items"}:
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step2_prune_queue",
            last_action=f"prune_queue:{decision.get('action', 'keep_queue')}",
        )

    requested = set(str(x) for x in decision.get("drop_identities", []) if x is not None)
    if not requested:
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step2_prune_queue",
            last_action="prune_queue:no_drops_selected",
        )

    valid_identities = {item.identity() for item in state.queue}
    if not requested <= valid_identities:
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step2_prune_queue",
            last_action="prune_queue:invalid_drop_identities_ignored",
        )

    kept: list[QueueItem] = []
    dropped: list[QueueItem] = []

    for item in state.queue:
        if item.identity() in requested:
            dropped.append(item)
        else:
            kept.append(item)

    for item in dropped:
        append_registry_event(
            RegistryEvent(
                time=current_time,
                event="queue_item_pruned",
                study_name=item.study_name or (item.proposal.new_study_name if item.proposal else None),
                reason=str(decision.get("reason", "llm_pruned")),
                payload={
                    "queue_source": item.source,
                    "queue_reason": item.reason,
                    "queue_identity": item.identity(),
                },
            )
        )

    kept.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
    return evolve_state(
        state,
        queue=tuple(kept),
        scratch=scratch,
        last_step="step2_prune_queue",
        last_action=f"prune_queue:dropped={len(dropped)}",
    )


def _build_queue_review_rows(state: TrainingSupervisorState) -> list[dict[str, Any]]:
    terminal_studies = {
        record.study_name
        for record in (
            list(state.registry_summary.completed_runs) + list(state.registry_summary.failed_or_killed_runs)
        )
    }
    active_study = state.active_run.study_name if state.active_run else None
    counts = Counter(item.identity() for item in state.queue)

    rows: list[dict[str, Any]] = []
    seen_so_far: set[str] = set()

    for item in state.queue:
        identity = item.identity()
        study_name = item.study_name or (item.proposal.new_study_name if item.proposal else None)
        review = {
            "identity": identity,
            "study_name": study_name,
            "queue_item": item.to_dict(),
            "duplicate_count": counts[identity],
            "is_duplicate_beyond_first": identity in seen_so_far,
            "is_active_study": study_name == active_study if study_name else False,
            "is_terminal_study": study_name in terminal_studies if study_name else False,
            "study_in_config_library": item.study_name in STUDIES if item.study_name else False,
            "blocked": _item_is_blocked(item, state.plan_view),
            "suggested_drop_reasons": [],
        }

        if review["is_duplicate_beyond_first"]:
            review["suggested_drop_reasons"].append("duplicate_beyond_first")
        if review["is_active_study"]:
            review["suggested_drop_reasons"].append("already_running")
        if review["is_terminal_study"]:
            review["suggested_drop_reasons"].append("already_terminal")
        if item.study_name and item.study_name not in STUDIES:
            review["suggested_drop_reasons"].append("missing_from_train_configs")
        if review["blocked"]:
            review["suggested_drop_reasons"].append("blocked_axis")

        rows.append(review)
        seen_so_far.add(identity)

    return rows


def _item_is_blocked(item: QueueItem, plan_view: Any) -> bool:
    if item.proposal is not None:
        meta = {
            "study_name": item.proposal.new_study_name,
            **item.proposal.changes,
        }
    elif item.study_name and item.study_name in STUDIES:
        meta = _cfg_to_meta(item.study_name, STUDIES[item.study_name].to_dict())
    else:
        return False

    pair_variant = meta.get("pair_variant")
    phi_variant = meta.get("phi_variant")

    if pair_variant is not None and _value_blocked("pair_variant", pair_variant, plan_view):
        return True
    if phi_variant is not None and _value_blocked("phi_variant", phi_variant, plan_view):
        return True
    return False


def _value_blocked(axis: str, value: Any, plan_view: Any) -> bool:
    blocked_axes = plan_view.blocked_axes
    currently_available = plan_view.currently_available

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
