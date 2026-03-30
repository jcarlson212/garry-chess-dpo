from __future__ import annotations

import os
import signal
import time

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.llm_policy import build_common_step_context, call_step_llm, record_step_decision
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import RegistryEvent, TrainingSupervisorState, evolve_state


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()

    if state.active_run is None or state.latest_observation is None:
        return evolve_state(
            state,
            last_step="step5_review_active_run",
            last_action="review_active_run:no_active_run",
        )

    min_budget = state.plan_view.screening_strategy.get("min_budget_before_comparison", {})
    min_epochs = int(min_budget.get("epochs", 1))
    min_steps = int(min_budget.get("steps", 200))
    min_budget_reached = _reached_min_budget(state, min_epochs=min_epochs, min_steps=min_steps)

    common = build_common_step_context(state)
    step_context = {
        **common,
        "policy_guardrails": {
            "never_kill_healthy_run_before_min_budget": bool(
                state.plan_view.scheduler_policy.get("never_kill_healthy_run_before_min_budget", True)
            ),
            "kill_if_stalled_minutes": state.plan_view.scheduler_policy.get("kill_if_stalled_minutes", 15),
            "kill_if_nan_or_traceback": bool(
                state.plan_view.scheduler_policy.get("kill_if_nan_or_traceback", True)
            ),
            "min_budget_before_comparison": {"epochs": min_epochs, "steps": min_steps},
            "min_budget_reached": min_budget_reached,
        },
    }

    llm_result = call_step_llm(
        state=state,
        step_name="step5_review_active_run",
        system_goal="Decide whether the currently running training job should continue or be terminated, based on health, progress, stall behavior, and policy.",
        allowed_actions=["keep_running", "terminate_run", "wait"],
        decision_schema_hint={
            "action": "terminate_run",
            "reason": "string",
            "reasoning_summary": "string",
            "terminate_mode": "graceful",
            "shared_context_updates": {},
        },
        step_context=step_context,
    )

    decision = dict(llm_result["decision"])
    scratch = record_step_decision(
        state,
        step_name="step5_review_active_run",
        llm_result=llm_result,
        step_context_snapshot=step_context,
    )

    if decision.get("action") != "terminate_run":
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step5_review_active_run",
            last_action=f"review_active_run:{decision.get('action', 'keep_running')}",
        )

    if _termination_blocked_by_policy(state, min_budget_reached=min_budget_reached):
        return evolve_state(
            state,
            scratch=scratch,
            last_step="step5_review_active_run",
            last_action="review_active_run:blocked_by_policy",
        )

    force = str(decision.get("terminate_mode", "graceful")).lower() == "force"
    _stop_run(state.active_run.pid, force=force)

    append_registry_event(
        RegistryEvent(
            time=current_time,
            event="run_killed",
            study_name=state.active_run.study_name,
            pid=state.active_run.pid,
            reason=str(decision.get("reason", "terminated_by_llm")),
            config_meta=_config_meta_from_active(state.active_run),
        )
    )

    scratch["terminated_run"] = state.active_run.to_dict()
    scratch["termination_reason"] = str(decision.get("reason", "terminated_by_llm"))

    return evolve_state(
        state,
        active_run=None,
        scratch=scratch,
        last_step="step5_review_active_run",
        last_action=f"review_active_run:killed:{scratch['termination_reason']}",
    )


def _reached_min_budget(state: TrainingSupervisorState, *, min_epochs: int, min_steps: int) -> bool:
    if state.latest_observation is None:
        return False
    latest_epoch = state.latest_observation.summary.latest_epoch or 0
    latest_step = state.latest_observation.summary.latest_global_step or 0
    return latest_epoch >= min_epochs or latest_step >= min_steps


def _termination_blocked_by_policy(state: TrainingSupervisorState, *, min_budget_reached: bool) -> bool:
    observation = state.latest_observation
    if observation is None:
        return False

    if observation.status in {"failed", "stalled"}:
        return False

    if bool(state.plan_view.scheduler_policy.get("never_kill_healthy_run_before_min_budget", True)):
        return not min_budget_reached

    return False


def _stop_run(pid: int, *, force: bool) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _config_meta_from_active(active: any) -> dict[str, object]:
    cfg = active.config
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    return {
        "study_name": active.study_name,
        "pair_variant": cfg.get("pair_variant"),
        "tau": cfg.get("tau"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "max_steps_per_epoch": cfg.get("max_steps_per_epoch"),
        "max_eval_batches": cfg.get("max_eval_batches"),
        "max_train_rows": cfg.get("max_train_rows"),
        "max_eval_rows": cfg.get("max_eval_rows"),
        "phi_variant": model.get("variant_name"),
        "embedding_dim": model.get("embedding_dim"),
    }
