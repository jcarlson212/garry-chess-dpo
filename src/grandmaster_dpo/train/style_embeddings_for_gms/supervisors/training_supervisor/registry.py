from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS, TrainingSupervisorPaths
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import append_jsonl, load_jsonl_with_fallback
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import RegistryEvent, RegistrySummary, StudyRecord


TERMINAL_EVENTS = {"run_finished", "run_failed", "run_killed", "run_canceled"}
FAILURE_EVENTS = {"run_failed", "run_killed", "run_canceled"}


def append_registry_event(
    event: RegistryEvent | Mapping[str, Any],
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> None:
    row = event.to_dict() if isinstance(event, RegistryEvent) else dict(event)
    append_jsonl(paths.run_registry_path, row)


def load_registry_events(paths: TrainingSupervisorPaths = PATHS) -> tuple[RegistryEvent, ...]:
    rows = load_jsonl_with_fallback(
        paths.run_registry_path,
        [paths.legacy_run_registry_path],
    )
    return tuple(RegistryEvent.from_dict(row) for row in rows if isinstance(row, Mapping))


def _completed_sort_key(record: StudyRecord) -> tuple[float, float, str]:
    loss = record.best_eval_loss if record.best_eval_loss is not None else float("inf")
    margin_key = -(record.best_eval_margin_cos if record.best_eval_margin_cos is not None else float("-inf"))
    return (loss, margin_key, record.study_name)


def build_registry_summary(paths: TrainingSupervisorPaths = PATHS) -> RegistrySummary:
    events = load_registry_events(paths=paths)
    by_study: dict[str, StudyRecord] = {}

    for event in events:
        if not event.study_name:
            continue

        current = by_study.get(event.study_name)
        if current is None:
            current = StudyRecord(study_name=event.study_name)
            by_study[event.study_name] = current

        updated = current

        if event.event == "run_started":
            updated = replace(updated, started=True)
        elif event.event == "run_finished":
            updated = replace(updated, finished=True)
            if event.best_eval_loss is not None:
                updated = replace(updated, best_eval_loss=event.best_eval_loss)
            if event.best_eval_margin_cos is not None:
                updated = replace(updated, best_eval_margin_cos=event.best_eval_margin_cos)
        elif event.event == "run_failed":
            updated = replace(updated, failed=True)
        elif event.event == "run_killed":
            updated = replace(updated, killed=True)
        elif event.event == "run_canceled":
            updated = replace(updated, canceled=True)

        if updated.config_meta is None and event.config_meta is not None:
            updated = replace(updated, config_meta=event.config_meta)

        updated = replace(updated, latest_event=event.event, latest_event_time=event.time)
        by_study[event.study_name] = updated

    all_records = tuple(by_study.values())
    completed_runs = tuple(sorted((r for r in all_records if r.finished), key=_completed_sort_key))
    failed_or_killed_runs = tuple(r for r in all_records if r.failed or r.killed or r.canceled)
    active_or_incomplete_runs = tuple(
        r for r in all_records if r.started and not (r.finished or r.failed or r.killed or r.canceled)
    )

    return RegistrySummary(
        num_registry_events=len(events),
        studies_seen=len(all_records),
        completed_runs=completed_runs,
        failed_or_killed_runs=failed_or_killed_runs,
        active_or_incomplete_runs=active_or_incomplete_runs,
        top_completed_runs=completed_runs[:5],
    )


def latest_event_for_study(
    study_name: str,
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> RegistryEvent | None:
    latest: RegistryEvent | None = None
    for event in load_registry_events(paths=paths):
        if event.study_name != study_name:
            continue
        if latest is None or event.time >= latest.time:
            latest = event
    return latest


def study_has_terminal_event(
    study_name: str,
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> bool:
    latest = latest_event_for_study(study_name, paths=paths)
    return latest.event in TERMINAL_EVENTS if latest is not None else False


def count_events_for_study(
    study_name: str,
    event_name: str,
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> int:
    total = 0
    for event in load_registry_events(paths=paths):
        if event.study_name == study_name and event.event == event_name:
            total += 1
    return total


def events_for_study(
    study_name: str,
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> tuple[RegistryEvent, ...]:
    return tuple(event for event in load_registry_events(paths=paths) if event.study_name == study_name)