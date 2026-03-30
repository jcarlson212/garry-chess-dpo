from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import psutil

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import append_handoff_event, load_json, load_jsonl_with_fallback
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.plan_view import build_training_context
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event, build_registry_summary, latest_event_for_study
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import (
    ActiveRun,
    RegistryEvent,
    RunObservation,
    SummaryDigest,
    TrainingSupervisorState,
    evolve_state,
)


SUMMARY_TAIL_LINES = 300
LOG_TAIL_BYTES = 20_000


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    current_time = now or time.time()
    active = state.active_run or recover_active_run()

    if active is None:
        registry_summary = build_registry_summary()
        _discover_terminal_runs_from_disk(
            current_time=current_time,
            existing_registry_summary=registry_summary,
        )
        registry_summary = build_registry_summary()
        scratch = dict(state.scratch)
        scratch["training_context"] = build_training_context(
            state.plan_view,
            registry_summary,
            queue=state.queue,
            active_run=None,
        )
        return evolve_state(
            state,
            active_run=None,
            latest_observation=None,
            registry_summary=registry_summary,
            last_step="step1_observe_runs",
            last_action="observe:no_active_run",
            scratch=scratch,
        )

    observation = observe_run(active, kill_if_stalled_minutes=_stall_minutes_from_plan(state))
    active_next = active if observation.status in {"running", "stalled"} else None

    _maybe_emit_terminal_registry_event(active, observation, current_time)

    registry_summary = build_registry_summary()
    scratch = dict(state.scratch)
    scratch["training_context"] = build_training_context(
        state.plan_view,
        registry_summary,
        queue=state.queue,
        active_run=active_next,
    )

    return evolve_state(
        state,
        active_run=active_next,
        latest_observation=observation,
        registry_summary=registry_summary,
        last_step="step1_observe_runs",
        last_action=f"observe:{observation.status}",
        scratch=scratch,
    )


def process_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def tail_text(path: str | Path, max_bytes: int = LOG_TAIL_BYTES) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("rb") as handle:
        size = p.stat().st_size
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="ignore")


def read_jsonl_tail(path: str | Path, max_lines: int = SUMMARY_TAIL_LINES) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def summarize_summary_file(summary_path: str | Path, *, min_time: float | None = None) -> SummaryDigest:
    rows = read_jsonl_tail(summary_path)
    if min_time is not None:
        rows = [
            row
            for row in rows
            if isinstance(row.get("time"), (int, float)) and float(row["time"]) >= min_time
        ]

    digest = SummaryDigest(
        summary_exists=Path(summary_path).exists(),
        event_count=len(rows),
        summary_tail="\n".join(json.dumps(row) for row in rows[-10:]),
    )

    best_eval_loss = None
    best_eval_margin = None
    latest_train_loss = None
    latest_pair_acc = None
    latest_samples_per_hour = None
    latest_global_step = None
    latest_epoch = None
    last_event_type = None
    last_event_time = None
    run_end_event = None
    timeout_stop_event = None

    for row in rows:
        event = row.get("event")
        if isinstance(event, str):
            last_event_type = event
            if isinstance(row.get("time"), (int, float)):
                last_event_time = float(row["time"])

        if event == "run_end":
            run_end_event = row
        elif event == "timeout_stop":
            timeout_stop_event = row
        elif event == "step_end":
            train_loss = row.get("train_loss")
            pair_acc = row.get("pair_acc")
            samples_per_hour = row.get("samples_per_hour_inst")
            global_step = row.get("global_step")
            epoch = row.get("epoch")

            latest_train_loss = float(train_loss) if isinstance(train_loss, (int, float)) else latest_train_loss
            latest_pair_acc = float(pair_acc) if isinstance(pair_acc, (int, float)) else latest_pair_acc
            latest_samples_per_hour = (
                float(samples_per_hour) if isinstance(samples_per_hour, (int, float)) else latest_samples_per_hour
            )
            latest_global_step = int(global_step) if isinstance(global_step, int) else latest_global_step
            latest_epoch = int(epoch) if isinstance(epoch, int) else latest_epoch

        elif event == "epoch_end":
            eval_block = row.get("eval") if isinstance(row.get("eval"), Mapping) else {}
            train_block = row.get("train") if isinstance(row.get("train"), Mapping) else {}
            eval_loss = eval_block.get("loss")
            margin_cos = eval_block.get("margin_cos")
            pair_acc = eval_block.get("pair_acc")
            samples_per_hour = row.get("samples_per_hour")
            epoch = row.get("epoch")

            if isinstance(eval_loss, (int, float)):
                eval_loss = float(eval_loss)
                if best_eval_loss is None or eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss

            if isinstance(margin_cos, (int, float)):
                margin_cos = float(margin_cos)
                if best_eval_margin is None or margin_cos > best_eval_margin:
                    best_eval_margin = margin_cos

            latest_train_loss = (
                float(train_block["loss"])
                if isinstance(train_block, Mapping) and isinstance(train_block.get("loss"), (int, float))
                else latest_train_loss
            )
            latest_pair_acc = float(pair_acc) if isinstance(pair_acc, (int, float)) else latest_pair_acc
            latest_samples_per_hour = (
                float(samples_per_hour) if isinstance(samples_per_hour, (int, float)) else latest_samples_per_hour
            )
            latest_epoch = int(epoch) if isinstance(epoch, int) else latest_epoch

    return SummaryDigest(
        summary_exists=digest.summary_exists,
        last_event_type=last_event_type,
        last_event_time=last_event_time,
        latest_global_step=latest_global_step,
        latest_epoch=latest_epoch,
        latest_train_loss=latest_train_loss,
        latest_pair_acc=latest_pair_acc,
        latest_samples_per_hour=latest_samples_per_hour,
        best_eval_loss=best_eval_loss,
        best_eval_margin_cos=best_eval_margin,
        event_count=len(rows),
        summary_tail=digest.summary_tail,
        run_end_event=run_end_event,
        timeout_stop_event=timeout_stop_event,
    )


def recover_active_run() -> ActiveRun | None:
    snapshot = load_json(
        PATHS.state_snapshot_path if PATHS.state_snapshot_path.exists() else PATHS.legacy_state_snapshot_path,
        {},
    )
    snapshot_active = snapshot.get("active_run")
    if isinstance(snapshot_active, Mapping):
        candidate = ActiveRun.from_dict(snapshot_active)
        digest = summarize_summary_file(candidate.summary_path, min_time=candidate.started_at)
        if digest.run_end_event is None and process_alive(candidate.pid):
            return ActiveRun(
                study_name=candidate.study_name,
                pid=candidate.pid,
                log_path=candidate.log_path,
                summary_path=candidate.summary_path,
                checkpoint_dir=candidate.checkpoint_dir,
                config=candidate.config,
                started_at=candidate.started_at,
                last_observed_at=time.time(),
                status="running",
                queue_source=candidate.queue_source,
                reason=candidate.reason,
            )

    rows = load_jsonl_with_fallback(
        PATHS.run_registry_path,
        [PATHS.legacy_run_registry_path],
    )
    last_started: Mapping[str, Any] | None = None
    terminal_by_study: dict[str, str] = {}

    for row in rows:
        study_name = row.get("study_name")
        if not study_name:
            continue
        event = row.get("event")
        if event == "run_started":
            last_started = row
        elif event in {"run_finished", "run_failed", "run_killed", "run_canceled"}:
            terminal_by_study[str(study_name)] = str(event)

    if last_started is None:
        return None

    study_name = last_started.get("study_name")
    pid = last_started.get("pid")
    if not isinstance(study_name, str) or not isinstance(pid, int):
        return None
    if study_name in terminal_by_study:
        return None
    if study_name not in STUDIES or not process_alive(pid):
        return None

    cfg = STUDIES[study_name]
    summary_path = str(cfg.summary_path())
    digest = summarize_summary_file(summary_path)
    if digest.run_end_event is not None:
        return None

    return ActiveRun(
        study_name=study_name,
        pid=pid,
        log_path=str(PATHS.runs_dir / study_name / "stdout.log"),
        summary_path=summary_path,
        checkpoint_dir=str(cfg.checkpoint_dir()),
        config=cfg.to_dict(),
        started_at=float(last_started.get("time", time.time())),
        last_observed_at=time.time(),
        status="running",
        queue_source=str(last_started.get("queue_source", "recovered")),
        reason=None,
    )


def observe_run(active: ActiveRun, *, kill_if_stalled_minutes: int) -> RunObservation:
    log_path = Path(active.log_path)
    log_exists = log_path.exists()
    log_tail = tail_text(log_path) if log_exists else ""
    last_log_mtime = log_path.stat().st_mtime if log_exists else None
    alive = process_alive(active.pid)
    digest = summarize_summary_file(active.summary_path, min_time=active.started_at)

    fatal_log = _contains_fatal_error(log_tail) or _contains_fatal_error(digest.summary_tail)

    if digest.run_end_event is not None:
        if _run_end_failed(digest.run_end_event) or fatal_log:
            status = "finished_failed"
            reason = "run_end_failed"
        else:
            status = "finished_success"
            reason = "run_end_success"
    elif digest.timeout_stop_event is not None:
        status = "finished_failed"
        reason = "timeout_stop"
    elif not alive:
        status = "failed" if fatal_log else "finished_failed"
        reason = "process_not_alive"
    elif fatal_log:
        status = "failed"
        reason = "fatal_log_pattern"
    elif _is_stalled(active, digest, log_exists, last_log_mtime, kill_if_stalled_minutes):
        status = "stalled"
        reason = "no_log_or_summary_progress"
    else:
        status = "running"
        reason = "healthy"

    return RunObservation(
        process_alive=alive,
        log_exists=log_exists,
        last_log_mtime=last_log_mtime,
        status=status,
        reason=reason,
        log_tail=log_tail[-4000:],
        summary=digest,
    )


def _stall_minutes_from_plan(state: TrainingSupervisorState) -> int:
    value = state.plan_view.scheduler_policy.get("kill_if_stalled_minutes", 15)
    try:
        return int(value)
    except Exception:
        return 15


def _contains_fatal_error(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "traceback",
            "nan",
            "cuda out of memory",
            "runtimeerror",
            "floatingpointerror",
            "assertionerror",
        ]
    )


def _run_end_failed(run_end_event: Mapping[str, Any]) -> bool:
    status = str(run_end_event.get("status", "")).lower()
    reason = str(run_end_event.get("reason", "")).lower()
    success = run_end_event.get("success")

    if success is False:
        return True
    if status in {"failed", "error"}:
        return True
    if "fail" in reason or "error" in reason:
        return True
    return False


def _is_stalled(
    active: ActiveRun,
    digest: SummaryDigest,
    log_exists: bool,
    last_log_mtime: float | None,
    stall_minutes: int,
) -> bool:
    cutoff_age = stall_minutes * 60
    now = time.time()
    freshest_time = max(
        [
            t
            for t in [digest.last_event_time, last_log_mtime, active.last_observed_at, active.started_at]
            if isinstance(t, (int, float))
        ],
        default=active.started_at,
    )
    return (now - freshest_time) >= cutoff_age


def _maybe_emit_terminal_registry_event(
    active: ActiveRun,
    observation: RunObservation,
    current_time: float,
) -> None:
    if observation.status not in {"finished_success", "finished_failed"}:
        return

    latest = latest_event_for_study(active.study_name)
    if latest is not None and latest.event in {"run_finished", "run_failed"}:
        return

    event_name = "run_finished" if observation.status == "finished_success" else "run_failed"
    append_registry_event(
        RegistryEvent(
            time=current_time,
            event=event_name,
            study_name=active.study_name,
            pid=active.pid,
            reason=observation.reason,
            config_meta=_config_meta_from_active(active),
            best_eval_loss=observation.summary.best_eval_loss,
            best_eval_margin_cos=observation.summary.best_eval_margin_cos,
        )
    )

    if observation.status == "finished_success":
        append_handoff_event(
            "checkpoint_ready_for_eval",
            created_at=current_time,
            payload={
                "study_name": active.study_name,
                "checkpoint_dir": active.checkpoint_dir,
                "summary_path": active.summary_path,
                "best_eval_loss": observation.summary.best_eval_loss,
                "best_eval_margin_cos": observation.summary.best_eval_margin_cos,
            },
        )


def _config_meta_from_active(active: ActiveRun) -> dict[str, Any]:
    model = active.config.get("model", {}) if isinstance(active.config.get("model"), Mapping) else {}
    return {
        "study_name": active.study_name,
        "pair_variant": active.config.get("pair_variant"),
        "tau": active.config.get("tau"),
        "lr": active.config.get("lr"),
        "batch_size": active.config.get("batch_size"),
        "epochs": active.config.get("epochs"),
        "max_steps_per_epoch": active.config.get("max_steps_per_epoch"),
        "max_eval_batches": active.config.get("max_eval_batches"),
        "max_train_rows": active.config.get("max_train_rows"),
        "max_eval_rows": active.config.get("max_eval_rows"),
        "phi_variant": model.get("variant_name"),
        "embedding_dim": model.get("embedding_dim"),
        "train_dir": active.config.get("train_dir"),
        "eval_dir": active.config.get("eval_dir"),
    }


def _config_meta_from_cfg(cfg: Any) -> dict[str, Any]:
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    model = cfg_dict.get("model", {}) if isinstance(cfg_dict.get("model"), Mapping) else {}
    return {
        "study_name": str(cfg_dict.get("study_name")),
        "pair_variant": cfg_dict.get("pair_variant"),
        "tau": cfg_dict.get("tau"),
        "lr": cfg_dict.get("lr"),
        "batch_size": cfg_dict.get("batch_size"),
        "epochs": cfg_dict.get("epochs"),
        "max_steps_per_epoch": cfg_dict.get("max_steps_per_epoch"),
        "max_eval_batches": cfg_dict.get("max_eval_batches"),
        "max_train_rows": cfg_dict.get("max_train_rows"),
        "max_eval_rows": cfg_dict.get("max_eval_rows"),
        "phi_variant": model.get("variant_name"),
        "embedding_dim": model.get("embedding_dim"),
        "train_dir": cfg_dict.get("train_dir"),
        "eval_dir": cfg_dict.get("eval_dir"),
    }


def _resolve_summary_path(summary_path: Path) -> Path:
    # `TrainConfig.summary_path()` uses relative paths from `train_configs.py`;
    # resolve them against repo root so running from any CWD works.
    if summary_path.is_absolute():
        return summary_path
    return (PATHS.project_root / summary_path).resolve()


def _discover_terminal_runs_from_disk(
    *,
    current_time: float,
    existing_registry_summary: Any,
) -> None:
    terminal_studies: set[str] = {r.study_name for r in existing_registry_summary.completed_runs}
    terminal_studies |= {r.study_name for r in existing_registry_summary.failed_or_killed_runs}

    for study_name, cfg in STUDIES.items():
        if study_name in terminal_studies:
            continue

        # Only attempt discovery if a summary file exists on disk.
        summary_path = _resolve_summary_path(cfg.summary_path())
        if not summary_path.exists():
            continue

        digest = summarize_summary_file(summary_path)
        if digest.run_end_event is not None:
            run_failed = _run_end_failed(digest.run_end_event)
            event_name = "run_failed" if run_failed else "run_finished"
            reason = str(digest.run_end_event.get("reason", "disk_detected_terminal"))
            append_registry_event(
                RegistryEvent(
                    time=current_time,
                    event=event_name,
                    study_name=study_name,
                    pid=None,
                    queue_source="disk_discovery",
                    reason=reason,
                    config_meta=_config_meta_from_cfg(cfg),
                    best_eval_loss=digest.best_eval_loss,
                    best_eval_margin_cos=digest.best_eval_margin_cos,
                )
            )
            terminal_studies.add(study_name)
            continue

        if digest.timeout_stop_event is not None:
            append_registry_event(
                RegistryEvent(
                    time=current_time,
                    event="run_failed",
                    study_name=study_name,
                    pid=None,
                    queue_source="disk_discovery",
                    reason="timeout_stop",
                    config_meta=_config_meta_from_cfg(cfg),
                    best_eval_loss=digest.best_eval_loss,
                    best_eval_margin_cos=digest.best_eval_margin_cos,
                )
            )
            terminal_studies.add(study_name)

