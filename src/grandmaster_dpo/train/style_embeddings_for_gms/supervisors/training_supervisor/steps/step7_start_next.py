from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES, make_config

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import upsert_materialized_study_config
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.registry import append_registry_event, study_has_terminal_event
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ActiveRun, QueueItem, RegistryEvent, RunObservation, SummaryDigest, TrainingSupervisorState, evolve_state


def run(state: TrainingSupervisorState, *, now: float | None = None) -> TrainingSupervisorState:
    if state.active_run is not None:
        return evolve_state(
            state,
            last_step="step7_start_next",
            last_action="start_next:already_running",
        )

    queue = list(state.queue)

    while queue:
        item = queue.pop(0)
        study_name = _materialize_queue_item(item)
        if study_name is None:
            append_registry_event(
                RegistryEvent(
                    time=time.time(),
                    event="drop_invalid_queue_item",
                    study_name=item.identity(),
                    reason="materialize_failed",
                )
            )
            continue

        if study_has_terminal_event(study_name):
            append_registry_event(
                RegistryEvent(
                    time=time.time(),
                    event="drop_invalid_queue_item",
                    study_name=study_name,
                    reason="already_terminal",
                )
            )
            continue

        active_run = _start_run(study_name, queue_source=item.source)
        initial_observation = RunObservation(
            process_alive=True,
            log_exists=Path(active_run.log_path).exists(),
            last_log_mtime=None,
            status="running",
            reason="started_by_supervisor",
            log_tail="",
            summary=SummaryDigest(),
        )

        return evolve_state(
            state,
            queue=tuple(queue),
            active_run=active_run,
            latest_observation=initial_observation,
            last_step="step7_start_next",
            last_action=f"start_next:started:{study_name}",
        )

    return evolve_state(
        state,
        queue=tuple(queue),
        last_step="step7_start_next",
        last_action="start_next:no_queue_item_started",
    )


def _materialize_queue_item(item: QueueItem) -> str | None:
    if item.study_name is not None:
        return item.study_name if item.study_name in STUDIES else None

    if item.proposal is None:
        return None

    proposal = item.proposal
    if proposal.base_study not in STUDIES:
        return None
    if proposal.new_study_name in STUDIES:
        return proposal.new_study_name

    base = STUDIES[proposal.base_study]
    changes = proposal.changes

    try:
        new_cfg = make_config(
            study_name=proposal.new_study_name,
            train_dir=changes.get("train_dir", base.train_dir),
            eval_dir=changes.get("eval_dir", base.eval_dir),
            pair_variant=changes.get("pair_variant", base.pair_variant),
            seed=changes.get("seed", base.seed),
            embedding_dim=changes.get("embedding_dim", base.model.embedding_dim),
            batch_size=changes.get("batch_size", base.batch_size),
            lr=changes.get("lr", base.lr),
            tau=changes.get("tau", base.tau),
            phi_variant=changes.get("phi_variant", base.model.variant_name),
            epochs=changes.get("epochs", base.epochs),
            max_steps_per_epoch=changes.get("max_steps_per_epoch", base.max_steps_per_epoch),
            max_eval_batches=changes.get("max_eval_batches", base.max_eval_batches),
            num_workers=changes.get("num_workers", base.num_workers),
            max_train_rows=changes.get("max_train_rows", base.max_train_rows),
            max_eval_rows=changes.get("max_eval_rows", base.max_eval_rows),
        )
    except Exception:
        return None

    STUDIES[proposal.new_study_name] = new_cfg
    upsert_materialized_study_config(proposal.new_study_name, new_cfg.to_dict(), paths=PATHS)
    return proposal.new_study_name


def _start_run(study_name: str, *, queue_source: str) -> ActiveRun:
    if study_name not in STUDIES:
        raise ValueError(f"Study not found: {study_name}")

    PATHS.ensure_dirs()
    run_dir = PATHS.runs_dir / study_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    log_handle = open(log_path, "a", buffering=1)

    cmd = [
        "python",
        "-m",
        "src.grandmaster_dpo.train.style_embeddings_for_gms.run_studies",
        "--studies",
        study_name,
    ]
    mat_path = PATHS.materialized_studies_path
    if mat_path.exists() and mat_path.stat().st_size > 0:
        cmd.extend(["--materialized-studies-json", str(mat_path.resolve())])

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        cwd=str(PATHS.project_root.resolve()),
    )

    cfg = STUDIES[study_name]
    active_run = ActiveRun(
        study_name=study_name,
        pid=proc.pid,
        log_path=str(log_path),
        summary_path=str(cfg.summary_path()),
        checkpoint_dir=str(cfg.checkpoint_dir()),
        config=cfg.to_dict(),
        started_at=time.time(),
        last_observed_at=time.time(),
        status="running",
        queue_source=queue_source,
        reason=None,
    )

    append_registry_event(
        RegistryEvent(
            time=time.time(),
            event="run_started",
            study_name=study_name,
            pid=proc.pid,
            queue_source=queue_source,
            config_meta=_cfg_to_meta(study_name, cfg.to_dict()),
        )
    )
    return active_run


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
        "max_train_rows": cfg.get("max_train_rows"),
        "max_eval_rows": cfg.get("max_eval_rows"),
        "phi_variant": model.get("variant_name"),
        "embedding_dim": model.get("embedding_dim"),
        "train_dir": cfg.get("train_dir"),
        "eval_dir": cfg.get("eval_dir"),
    }
