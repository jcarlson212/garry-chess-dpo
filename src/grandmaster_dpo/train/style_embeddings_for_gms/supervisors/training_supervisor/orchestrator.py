from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT: Path | None = None
for _parent in _THIS_FILE.parents:
    if _parent.name == "src":
        _SRC_ROOT = _parent
        break

# Ensure we import this repo's `src/grandmaster_dpo/...` rather than an
# installed `grandmaster_dpo` package from `site-packages`.
if _SRC_ROOT is not None:
    _src_root_str = str(_SRC_ROOT)
    if _src_root_str not in sys.path:
        sys.path.insert(0, _src_root_str)

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import load_proposed_studies, load_queue, load_snapshot, persist_state_artifacts
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.plan_view import (
    build_training_plan_view_from_disk,
    validate_experiment_plan_loaded,
)
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event, build_registry_summary
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ActiveRun, ProposalSpec, QueueItem, TrainingSupervisorState, evolve_state

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.steps import (
    step1_observe_runs,
    step2_prune_queue,
    step3_propose_trials,
    step4_enqueue_trials,
    step5_review_active_run,
    step6_requeue_canceled,
    step7_start_next,
)


DEFAULT_POLL_SECONDS = int(os.environ.get("TRAINING_SUPERVISOR_POLL_SECONDS", "300"))


StepFn = Callable[[TrainingSupervisorState], TrainingSupervisorState]


def build_initial_state() -> TrainingSupervisorState:
    """
    Build initial in-memory state using:
    - shared experiment plan from disk
    - registry summary from disk
    - queue/proposed studies from disk
    - prior snapshot for active_run / loop counters / scratch continuity
    """
    print("Loading initial state from disk...")
    PATHS.ensure_dirs()

    validate_experiment_plan_loaded(paths=PATHS)
    print(f"[training_supervisor] project_root={PATHS.project_root.resolve()}")
    print(f"[training_supervisor] experiment_plan={PATHS.shared_experiment_plan_path.resolve()}")
    print(f"[training_supervisor] materialized_studies={PATHS.materialized_studies_path.resolve()}")

    snapshot = load_snapshot(paths=PATHS)
    plan_view = build_training_plan_view_from_disk(paths=PATHS)
    registry_summary = build_registry_summary(paths=PATHS)
    queue = load_queue(paths=PATHS)
    proposed_studies = load_proposed_studies(paths=PATHS)

    active_run = None
    snapshot_active = snapshot.get("active_run")
    if isinstance(snapshot_active, dict):
        try:
            active_run = ActiveRun.from_dict(snapshot_active)
        except Exception:
            active_run = None

    loop_iteration = int(snapshot.get("loop_iteration", 0))
    last_step = snapshot.get("last_step")
    last_action = snapshot.get("last_action")
    scratch = dict(snapshot.get("scratch", {})) if isinstance(snapshot.get("scratch"), dict) else {}

    # Ensure long-lived keys exist.
    scratch.setdefault("llm_calls", 0)
    scratch.setdefault("llm_trace", [])
    scratch.setdefault("shared_step_context", {})
    scratch.setdefault("step_decisions", {})

    return TrainingSupervisorState(
        plan_view=plan_view,
        registry_summary=registry_summary,
        queue=queue,
        proposed_studies=proposed_studies,
        active_run=active_run,
        latest_observation=None,
        last_step=str(last_step) if last_step is not None else None,
        last_action=str(last_action) if last_action is not None else None,
        loop_iteration=loop_iteration,
        scratch=scratch,
    )


def refresh_loop_inputs(state: TrainingSupervisorState) -> TrainingSupervisorState:
    """
    Reload external inputs at the start of every loop so manual file edits
    and newly appended registry/proposal entries are visible.
    """
    refreshed = evolve_state(
        state,
        plan_view=build_training_plan_view_from_disk(paths=PATHS),
        registry_summary=build_registry_summary(paths=PATHS),
        queue=load_queue(paths=PATHS),
        proposed_studies=load_proposed_studies(paths=PATHS),
    )
    return refreshed


def prepare_new_loop(state: TrainingSupervisorState) -> TrainingSupervisorState:
    """
    Start a fresh per-loop decision context while preserving long-lived trace.
    Steps 2-6 can pass context to later steps via shared_step_context and
    step_decisions during this loop.
    """
    scratch = dict(state.scratch)

    llm_trace = list(scratch.get("llm_trace", []))
    llm_calls = int(scratch.get("llm_calls", 0))

    scratch["llm_trace"] = llm_trace
    scratch["llm_calls"] = llm_calls
    scratch["shared_step_context"] = {}
    scratch["step_decisions"] = {}
    scratch.pop("terminated_run", None)
    scratch.pop("termination_reason", None)
    scratch.pop("last_error", None)

    return evolve_state(
        state,
        loop_iteration=state.loop_iteration + 1,
        scratch=scratch,
        last_step="loop_start",
        last_action="loop_start",
    )


def get_poll_seconds(state: TrainingSupervisorState, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    value = state.plan_view.scheduler_policy.get("poll_seconds", DEFAULT_POLL_SECONDS)
    try:
        return max(1, int(value))
    except Exception:
        return DEFAULT_POLL_SECONDS


def persist_checkpoint(state: TrainingSupervisorState, *, now: float | None = None) -> None:
    persist_state_artifacts(state, now=now or time.time(), paths=PATHS)


def run_single_step(
    state: TrainingSupervisorState,
    step_name: str,
    step_fn: StepFn,
) -> TrainingSupervisorState:
    print(f"Running step: {step_name}...")
    next_state = step_fn(state)
    persist_checkpoint(next_state)
    return next_state


def run_one_loop(state: TrainingSupervisorState) -> TrainingSupervisorState:
    state = refresh_loop_inputs(state)
    state = prepare_new_loop(state)
    persist_checkpoint(state)

    ordered_steps: list[tuple[str, StepFn]] = [
        ("step1_observe_runs", step1_observe_runs.run),
        ("step3_propose_trials", step3_propose_trials.run),
        ("step4_enqueue_trials", step4_enqueue_trials.run),
        ("step5_review_active_run", step5_review_active_run.run),
        ("step6_requeue_canceled", step6_requeue_canceled.run),
        ("step7_start_next", step7_start_next.run),
    ]

    for step_name, step_fn in ordered_steps:
        print(f"Starting step: {step_name}...")
        state = run_single_step(state, step_name, step_fn)
    
    print(f"Completed loop iteration. State summary: {state}")
    return state


def handle_loop_exception(state: TrainingSupervisorState, exc: BaseException) -> TrainingSupervisorState:
    tb = traceback.format_exc()

    append_registry_event(
        {
            "time": time.time(),
            "event": "supervisor_loop_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "loop_iteration": state.loop_iteration,
            "last_step": state.last_step,
            "last_action": state.last_action,
        },
        paths=PATHS,
    )

    scratch = dict(state.scratch)
    scratch["last_error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": tb,
        "loop_iteration": state.loop_iteration,
        "last_step": state.last_step,
        "last_action": state.last_action,
        "time": time.time(),
    }

    errored_state = evolve_state(
        state,
        scratch=scratch,
        last_step="loop_exception",
        last_action=f"loop_exception:{type(exc).__name__}",
    )
    persist_checkpoint(errored_state)
    return errored_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training supervisor orchestrator")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single supervisor loop and exit.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=None,
        help="Override poll interval between loops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = build_initial_state()
    persist_checkpoint(state)

    try:
        if args.once:
            state = run_one_loop(state)
            persist_checkpoint(state)
            return

        while True:
            try:
                state = run_one_loop(state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                state = handle_loop_exception(state, exc)

            sleep_for = get_poll_seconds(state, override=args.poll_seconds)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        append_registry_event(
            {
                "time": time.time(),
                "event": "supervisor_stopped",
                "reason": "keyboard_interrupt",
                "loop_iteration": state.loop_iteration,
                "last_step": state.last_step,
                "last_action": state.last_action,
            },
            paths=PATHS,
        )

        scratch = dict(state.scratch)
        scratch["stopped_by"] = "keyboard_interrupt"
        scratch["stopped_at"] = time.time()

        stopped_state = evolve_state(
            state,
            scratch=scratch,
            last_step="supervisor_stopped",
            last_action="keyboard_interrupt",
        )
        persist_checkpoint(stopped_state)
        return


if __name__ == "__main__":
    main()
