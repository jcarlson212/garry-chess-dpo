from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal, Mapping


JsonDict = dict[str, Any]


def _filter_known_fields(cls: type, raw: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


@dataclass(frozen=True)
class ProposalSpec:
    base_study: str
    new_study_name: str
    changes: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProposalSpec":
        payload = _filter_known_fields(cls, raw)
        payload["base_study"] = str(payload["base_study"])
        payload["new_study_name"] = str(payload["new_study_name"])
        payload["changes"] = dict(payload.get("changes", {}))
        return cls(**payload)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class QueueItem:
    priority: int
    created_at: float
    source: str
    reason: str
    study_name: str | None = None
    proposal: ProposalSpec | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "QueueItem":
        proposal_raw = raw.get("proposal")
        proposal = ProposalSpec.from_dict(proposal_raw) if isinstance(proposal_raw, Mapping) else None
        return cls(
            priority=int(raw.get("priority", 0)),
            created_at=float(raw.get("created_at", 0.0)),
            source=str(raw.get("source", "system")),
            reason=str(raw.get("reason", "")),
            study_name=str(raw["study_name"]) if raw.get("study_name") is not None else None,
            proposal=proposal,
        )

    def identity(self) -> str:
        if self.study_name:
            return self.study_name
        if self.proposal:
            return self.proposal.new_study_name
        return "unknown"

    def to_dict(self) -> JsonDict:
        out = asdict(self)
        if self.study_name is None:
            out.pop("study_name", None)
        if self.proposal is None:
            out.pop("proposal", None)
        return out


@dataclass(frozen=True)
class ActiveRun:
    study_name: str
    pid: int
    log_path: str
    summary_path: str
    checkpoint_dir: str
    config: JsonDict
    started_at: float
    last_observed_at: float
    status: str
    queue_source: str = "queue"
    reason: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActiveRun":
        payload = _filter_known_fields(cls, raw)
        payload["study_name"] = str(payload["study_name"])
        payload["pid"] = int(payload["pid"])
        payload["log_path"] = str(payload["log_path"])
        payload["summary_path"] = str(payload["summary_path"])
        payload["checkpoint_dir"] = str(payload["checkpoint_dir"])
        payload["config"] = dict(payload.get("config", {}))
        payload["started_at"] = float(payload.get("started_at", 0.0))
        payload["last_observed_at"] = float(payload.get("last_observed_at", 0.0))
        payload["status"] = str(payload.get("status", "unknown"))
        payload["queue_source"] = str(payload.get("queue_source", "queue"))
        return cls(**payload)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class SummaryDigest:
    summary_exists: bool = False
    last_event_type: str | None = None
    last_event_time: float | None = None
    latest_global_step: int | None = None
    latest_epoch: int | None = None
    latest_train_loss: float | None = None
    latest_pair_acc: float | None = None
    latest_samples_per_hour: float | None = None
    best_eval_loss: float | None = None
    best_eval_margin_cos: float | None = None
    event_count: int = 0
    summary_tail: str = ""
    run_end_event: JsonDict | None = None
    timeout_stop_event: JsonDict | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SummaryDigest":
        payload = _filter_known_fields(cls, raw)
        return cls(**payload)

    def to_dict(self) -> JsonDict:
        return asdict(self)


ObservationStatus = Literal[
    "idle",
    "running",
    "finished_success",
    "finished_failed",
    "stalled",
    "failed",
]


@dataclass(frozen=True)
class RunObservation:
    process_alive: bool
    log_exists: bool
    last_log_mtime: float | None
    status: ObservationStatus
    reason: str | None
    log_tail: str = ""
    summary: SummaryDigest = field(default_factory=SummaryDigest)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RunObservation":
        summary_raw = raw.get("summary", {})
        return cls(
            process_alive=bool(raw.get("process_alive", False)),
            log_exists=bool(raw.get("log_exists", False)),
            last_log_mtime=(
                float(raw["last_log_mtime"]) if raw.get("last_log_mtime") is not None else None
            ),
            status=str(raw.get("status", "idle")),  # type: ignore[arg-type]
            reason=str(raw["reason"]) if raw.get("reason") is not None else None,
            log_tail=str(raw.get("log_tail", "")),
            summary=SummaryDigest.from_dict(summary_raw) if isinstance(summary_raw, Mapping) else SummaryDigest(),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class RegistryEvent:
    time: float
    event: str
    study_name: str | None = None
    reason: str | None = None
    pid: int | None = None
    queue_source: str | None = None
    config_meta: JsonDict | None = None
    best_eval_loss: float | None = None
    best_eval_margin_cos: float | None = None
    payload: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RegistryEvent":
        known = {
            "time",
            "event",
            "study_name",
            "reason",
            "pid",
            "queue_source",
            "config_meta",
            "best_eval_loss",
            "best_eval_margin_cos",
        }
        payload = {k: v for k, v in raw.items() if k not in known}
        return cls(
            time=float(raw.get("time", 0.0)),
            event=str(raw.get("event", "")),
            study_name=str(raw["study_name"]) if raw.get("study_name") is not None else None,
            reason=str(raw["reason"]) if raw.get("reason") is not None else None,
            pid=int(raw["pid"]) if raw.get("pid") is not None else None,
            queue_source=str(raw["queue_source"]) if raw.get("queue_source") is not None else None,
            config_meta=dict(raw.get("config_meta", {})) if raw.get("config_meta") is not None else None,
            best_eval_loss=(
                float(raw["best_eval_loss"]) if raw.get("best_eval_loss") is not None else None
            ),
            best_eval_margin_cos=(
                float(raw["best_eval_margin_cos"]) if raw.get("best_eval_margin_cos") is not None else None
            ),
            payload=payload,
        )

    def to_dict(self) -> JsonDict:
        base = asdict(self)
        payload = base.pop("payload", {})
        return {**base, **payload}


@dataclass(frozen=True)
class StudyRecord:
    study_name: str
    started: bool = False
    finished: bool = False
    failed: bool = False
    killed: bool = False
    canceled: bool = False
    latest_event: str | None = None
    latest_event_time: float | None = None
    config_meta: JsonDict | None = None
    best_eval_loss: float | None = None
    best_eval_margin_cos: float | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class RegistrySummary:
    num_registry_events: int = 0
    studies_seen: int = 0
    completed_runs: tuple[StudyRecord, ...] = ()
    failed_or_killed_runs: tuple[StudyRecord, ...] = ()
    active_or_incomplete_runs: tuple[StudyRecord, ...] = ()
    top_completed_runs: tuple[StudyRecord, ...] = ()

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class AblationSpec:
    axis: str
    required_for: tuple[str, ...]
    values: tuple[Any, ...]
    status: str
    notes: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class CoverageStatus:
    axis: str
    required: tuple[Any, ...]
    completed: tuple[Any, ...]
    queued: tuple[Any, ...]
    running: tuple[Any, ...]
    blocked: tuple[Any, ...]
    remaining: tuple[Any, ...]

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPlanView:
    experiment_family: str
    experiment_version: str
    description: str
    training_goals: tuple[str, ...]
    downstream_goals: tuple[str, ...]
    required_ablations: dict[str, AblationSpec]
    allowed_axes: JsonDict
    blocked_axes: JsonDict
    currently_available: JsonDict
    scheduler_policy: JsonDict
    screening_strategy: JsonDict
    promotion_rules: JsonDict
    downstream_awareness: JsonDict

    def to_dict(self) -> JsonDict:
        return {
            "experiment_family": self.experiment_family,
            "experiment_version": self.experiment_version,
            "description": self.description,
            "training_goals": list(self.training_goals),
            "downstream_goals": list(self.downstream_goals),
            "required_ablations": {k: v.to_dict() for k, v in self.required_ablations.items()},
            "allowed_axes": self.allowed_axes,
            "blocked_axes": self.blocked_axes,
            "currently_available": self.currently_available,
            "scheduler_policy": self.scheduler_policy,
            "screening_strategy": self.screening_strategy,
            "promotion_rules": self.promotion_rules,
            "downstream_awareness": self.downstream_awareness,
        }


@dataclass(frozen=True)
class StateSnapshot:
    time: float
    loop_iteration: int
    last_step: str | None
    last_action: str | None
    active_run: ActiveRun | None
    latest_status: str | None
    latest_reason: str | None
    queue: tuple[QueueItem, ...] = ()
    proposed_studies: tuple[ProposalSpec, ...] = ()
    scratch: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "time": self.time,
            "loop_iteration": self.loop_iteration,
            "last_step": self.last_step,
            "last_action": self.last_action,
            "active_run": self.active_run.to_dict() if self.active_run else None,
            "latest_status": self.latest_status,
            "latest_reason": self.latest_reason,
            "queue": [q.to_dict() for q in self.queue],
            "proposed_studies": [p.to_dict() for p in self.proposed_studies],
            "scratch": self.scratch,
        }


@dataclass(frozen=True)
class TrainingSupervisorState:
    plan_view: TrainingPlanView
    registry_summary: RegistrySummary
    queue: tuple[QueueItem, ...] = ()
    proposed_studies: tuple[ProposalSpec, ...] = ()
    active_run: ActiveRun | None = None
    latest_observation: RunObservation | None = None
    last_step: str | None = None
    last_action: str | None = None
    loop_iteration: int = 0
    scratch: JsonDict = field(default_factory=dict)

    def to_snapshot(self, now: float) -> StateSnapshot:
        return StateSnapshot(
            time=now,
            loop_iteration=self.loop_iteration,
            last_step=self.last_step,
            last_action=self.last_action,
            active_run=self.active_run,
            latest_status=self.latest_observation.status if self.latest_observation else None,
            latest_reason=self.latest_observation.reason if self.latest_observation else None,
            queue=self.queue,
            proposed_studies=self.proposed_studies,
            scratch=self.scratch,
        )


def evolve_state(state: TrainingSupervisorState, **changes: Any) -> TrainingSupervisorState:
    return replace(state, **changes)


