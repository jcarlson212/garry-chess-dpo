from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS, TrainingSupervisorPaths
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import ProposalSpec, QueueItem, StateSnapshot, TrainingSupervisorState


T = TypeVar("T")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return [_serialize(v) for v in value]
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_json(path: Path, default: T) -> T:
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return default


def load_json_with_fallback(primary: Path, fallbacks: list[Path], default: T) -> T:
    if primary.exists():
        return load_json(primary, default)
    for path in fallbacks:
        if path.exists():
            return load_json(path, default)
    return default


def save_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(_serialize(payload), indent=2, default=str) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_jsonl_with_fallback(primary: Path, fallbacks: list[Path]) -> list[dict[str, Any]]:
    if primary.exists():
        return load_jsonl(primary)
    for path in fallbacks:
        if path.exists():
            return load_jsonl(path)
    return []


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), default=str) + "\n")


def load_queue(paths: TrainingSupervisorPaths = PATHS) -> tuple[QueueItem, ...]:
    raw = load_json_with_fallback(
        paths.queue_path,
        [paths.legacy_queue_path],
        [],
    )
    if not isinstance(raw, list):
        return ()
    items = [QueueItem.from_dict(x) for x in raw if isinstance(x, Mapping)]
    items.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
    return tuple(items)


def save_queue(queue: Iterable[QueueItem], paths: TrainingSupervisorPaths = PATHS) -> None:
    items = list(queue)
    items.sort(key=lambda q: (-q.priority, q.created_at, q.identity()))
    save_json(paths.queue_path, [item.to_dict() for item in items])


def load_proposed_studies(paths: TrainingSupervisorPaths = PATHS) -> tuple[ProposalSpec, ...]:
    rows = load_jsonl_with_fallback(
        paths.proposed_studies_path,
        [paths.legacy_proposed_studies_path],
    )
    out: list[ProposalSpec] = []
    seen: set[str] = set()

    for row in rows:
        proposal_raw = row.get("proposal", row)
        if not isinstance(proposal_raw, Mapping):
            continue
        try:
            proposal = ProposalSpec.from_dict(proposal_raw)
        except Exception:
            continue
        if proposal.new_study_name in seen:
            continue
        seen.add(proposal.new_study_name)
        out.append(proposal)

    return tuple(out)


def append_proposed_study(
    proposal: ProposalSpec,
    *,
    reason: str,
    source: str,
    created_at: float,
    paths: TrainingSupervisorPaths = PATHS,
) -> None:
    append_jsonl(
        paths.proposed_studies_path,
        {
            "created_at": created_at,
            "source": source,
            "reason": reason,
            "proposal": proposal.to_dict(),
        },
    )


def append_handoff_event(
    event_type: str,
    *,
    created_at: float,
    payload: Mapping[str, Any],
    paths: TrainingSupervisorPaths = PATHS,
) -> None:
    append_jsonl(
        paths.handoff_events_path,
        {
            "time": created_at,
            "event": event_type,
            **dict(payload),
        },
    )


def load_snapshot(paths: TrainingSupervisorPaths = PATHS) -> dict[str, Any]:
    raw = load_json_with_fallback(
        paths.state_snapshot_path,
        [paths.legacy_state_snapshot_path],
        {},
    )
    return raw if isinstance(raw, dict) else {}


def save_snapshot(snapshot: StateSnapshot, paths: TrainingSupervisorPaths = PATHS) -> None:
    save_json(paths.state_snapshot_path, snapshot.to_dict())


def upsert_materialized_study_config(
    study_name: str,
    cfg_dict: Mapping[str, Any],
    *,
    paths: TrainingSupervisorPaths = PATHS,
) -> None:
    """
    Persist a study config so the training subprocess can load it via
    run_studies --materialized-studies-json (parent-only STUDIES mutations are invisible to children).
    """
    paths.ensure_dirs()
    target = paths.materialized_studies_path
    current = load_json(target, {})
    if not isinstance(current, dict):
        current = {}
    current[str(study_name)] = dict(cfg_dict)
    save_json(target, current)


def persist_state_artifacts(
    state: TrainingSupervisorState,
    *,
    now: float,
    paths: TrainingSupervisorPaths = PATHS,
) -> None:
    paths.ensure_dirs()
    save_queue(state.queue, paths=paths)
    save_snapshot(state.to_snapshot(now=now), paths=paths)