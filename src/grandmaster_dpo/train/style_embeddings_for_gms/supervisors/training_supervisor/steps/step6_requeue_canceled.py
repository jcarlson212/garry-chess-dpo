from __future__ import annotations

import time

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.llm_policy import build_common_step_context, call_step_llm, record_step_decision
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event, count_events_for_study
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ActiveRun, QueueItem, RegistryEvent, TrainingSupervisorState, evolve_state


MAX_REQUEUE_ATTEMPTS = 1


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()
    terminated_raw = state.scratch.get("terminated_run")
    termination_reason = state.scratch.get("termination_reason")

    if not isinstance(terminated_raw, dict):
        return evolve_state(
            state,
            last_step="step6_requeue_canceled",
            last_action="requeue_canceled:none",
        )

    terminated = ActiveRun.from_dict(terminated_raw)
    prior_requeues = count_events_for_study(terminated.study_name, "run_requeued")

    common = build_common_step_context(state)
    step_context = {
        **common,
        "terminated_run": terminated.to_dict(),
        "termination_reason": termination_reason,
        "prior_requeues": prior_requeues,
        "max_requeue_attempts": MAX_REQUEUE_ATTEMPTS,
    }

    llm_result = call_step_llm(
        state=state,
        step_name="step6_requeue_canceled",
        system_goal="Decide whether a recently terminated training study should be retried later by putting it back into the training queue.",
        allowed_actions=["do_not_requeue", "requeue_trial", "wait"],
        decision_schema_hint={
            "action": "requeue_trial",
            "reason": "string",
            "reasoning_summary": "string",
            "priority": 5,
            "shared_context_updates": {},
        },
        step_context=step_context,
    )

    decision = dict(llm_result["decision"])
    scratch = record_step_decision(
        state,
        step_name="step6_requeue_canceled",
        llm_result=llm_result,
        step_context_snapshot=step_context,
    )

    if prior_requeues >= MAX_REQUEUE_ATTEMPTS:
        decision_action = "do_not_requeue"
    else:
        decision_action = decision.get("action")

    if decision_action != "requeue_trial":
        scratch.pop("terminated_run", None)
        scratch.pop("termination_reason", None)
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step6_requeue_canceled",
            last_action=f"requeue_canceled:{decision_action or 'do_not_requeue'}",
        )

    queue = list(state.queue)
    if all(item.identity() != terminated.study_name for item in queue):
        priority = int(decision.get("priority", 5))
        queue.append(
            QueueItem(
                study_name=terminated.study_name,
                priority=priority,
                created_at=current_time,
                source="llm_requeue",
                reason=str(decision.get("reason", "retry_requested")),
            )
        )
        append_registry_event(
            RegistryEvent(
                time=current_time,
                event="run_requeued",
                study_name=terminated.study_name,
                reason=str(decision.get("reason", "retry_requested")),
                payload={"priority": priority},
            )
        )

    queue.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
    scratch.pop("terminated_run", None)
    scratch.pop("termination_reason", None)

    return evolve_state(
        state,
        queue=tuple(queue),
        scratch=scratch,
        last_step="step6_requeue_canceled",
        last_action=f"requeue_canceled:yes:{terminated.study_name}",
    )

