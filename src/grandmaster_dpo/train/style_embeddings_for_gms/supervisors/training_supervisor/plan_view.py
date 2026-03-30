from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.paths import PATHS, TrainingSupervisorPaths
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.persistence import load_json
from grandmaster_dpo.train.style_embeddings_for_gms.supervisors.training_supervisor.supervisor_types import AblationSpec, ActiveRun, CoverageStatus, QueueItem, RegistrySummary, TrainingPlanView


def load_full_experiment_plan(paths: TrainingSupervisorPaths = PATHS) -> dict[str, Any]:
    raw = load_json(paths.shared_experiment_plan_path, {})
    return raw if isinstance(raw, dict) else {}


def validate_experiment_plan_loaded(paths: TrainingSupervisorPaths = PATHS) -> None:
    """
    Fail fast if experiment_plan.json is missing or empty so scheduling rules
    (blocked axes, coverage) are not silently disabled.
    """
    if not paths.shared_experiment_plan_path.exists():
        raise FileNotFoundError(
            f"Experiment plan not found. Create or symlink experiment_plan.json at:\n  {paths.shared_experiment_plan_path}"
        )
    raw = load_json(paths.shared_experiment_plan_path, {})
    if not raw:
        raise ValueError(
            f"Experiment plan file is empty or invalid JSON: {paths.shared_experiment_plan_path}"
        )
    fam = str(raw.get("experiment_family", "")).strip()
    if not fam or fam == "unknown":
        raise ValueError(
            f"Experiment plan must set a non-empty 'experiment_family' (not 'unknown'): {paths.shared_experiment_plan_path}"
        )


def build_training_plan_view(full_plan: Mapping[str, Any]) -> TrainingPlanView:
    goals = tuple(str(x) for x in full_plan.get("goals", []))
    training_goals = tuple(goal for goal in goals if not _looks_downstream_only_goal(goal))
    downstream_goals = tuple(goal for goal in goals if _looks_downstream_only_goal(goal))

    required_ablations_raw = full_plan.get("required_ablations", {})
    required_ablations: dict[str, AblationSpec] = {}

    if isinstance(required_ablations_raw, Mapping):
        for axis, spec in required_ablations_raw.items():
            if not isinstance(spec, Mapping):
                continue
            required_ablations[str(axis)] = AblationSpec(
                axis=str(axis),
                required_for=tuple(str(x) for x in spec.get("required_for", [])),
                values=tuple(spec.get("values", [])),
                status=str(spec.get("status", "unknown")),
                notes=str(spec["notes"]) if spec.get("notes") is not None else None,
            )

    allowed_axes = dict(full_plan.get("allowed_axes", {}))
    stage_values = allowed_axes.get("stage")
    if isinstance(stage_values, list):
        allowed_axes["stage"] = [x for x in stage_values if x != "analysis"]

    downstream_awareness = {
        "paper_goals": list(full_plan.get("paper_goals", [])),
        "required_metrics": list(full_plan.get("required_metrics", [])),
        "required_plots": list(full_plan.get("required_plots", [])),
        "analysis_jobs": list(full_plan.get("analysis_jobs", [])),
        "analysis_prerequisites_ready": list(
            (full_plan.get("currently_available", {}) or {}).get("analysis_prerequisites_ready", [])
        ),
        "analysis_prerequisites_pending": list(
            (full_plan.get("currently_available", {}) or {}).get("analysis_prerequisites_pending", [])
        ),
    }

    return TrainingPlanView(
        experiment_family=str(full_plan.get("experiment_family", "unknown")),
        experiment_version=str(full_plan.get("experiment_version", "unknown")),
        description=str(full_plan.get("description", "")),
        training_goals=training_goals,
        downstream_goals=downstream_goals,
        required_ablations=required_ablations,
        allowed_axes=allowed_axes,
        blocked_axes=dict(full_plan.get("blocked_axes", {})),
        currently_available=dict(full_plan.get("currently_available", {})),
        scheduler_policy=dict(full_plan.get("scheduler_policy", {})),
        screening_strategy=dict(full_plan.get("screening_strategy", {})),
        promotion_rules=dict(full_plan.get("promotion_rules", {})),
        downstream_awareness=downstream_awareness,
    )


def build_training_plan_view_from_disk(paths: TrainingSupervisorPaths = PATHS) -> TrainingPlanView:
    return build_training_plan_view(load_full_experiment_plan(paths=paths))


def compute_training_coverage(
    plan_view: TrainingPlanView,
    registry_summary: RegistrySummary,
    *,
    queue: Iterable[QueueItem] = (),
    active_run: ActiveRun | None = None,
) -> dict[str, CoverageStatus]:
    completed_meta = [record.config_meta or {"study_name": record.study_name} for record in registry_summary.completed_runs]
    queued_meta = [_queue_item_meta(item) for item in queue]
    running_meta = [active_run.config] if active_run is not None else []

    coverage: dict[str, CoverageStatus] = {}

    for axis, spec in plan_view.required_ablations.items():
        required = tuple(spec.values)
        completed = _stable_unique(
            value
            for value in (_extract_axis_value(meta, axis) for meta in completed_meta)
            if _value_in_required(value, required)
        )
        queued = _stable_unique(
            value
            for value in (_extract_axis_value(meta, axis) for meta in queued_meta)
            if _value_in_required(value, required) and value not in completed
        )
        running = _stable_unique(
            value
            for value in (_extract_axis_value(meta, axis) for meta in running_meta)
            if _value_in_required(value, required) and value not in completed
        )
        blocked = tuple(value for value in required if _is_blocked_value(axis, value, plan_view))
        remaining = tuple(
            value
            for value in required
            if value not in completed and value not in queued and value not in running and value not in blocked
        )

        coverage[axis] = CoverageStatus(
            axis=axis,
            required=required,
            completed=completed,
            queued=queued,
            running=running,
            blocked=blocked,
            remaining=remaining,
        )

    return coverage


def build_training_context(
    plan_view: TrainingPlanView,
    registry_summary: RegistrySummary,
    *,
    queue: Iterable[QueueItem] = (),
    active_run: ActiveRun | None = None,
) -> dict[str, Any]:
    coverage = compute_training_coverage(
        plan_view,
        registry_summary,
        queue=queue,
        active_run=active_run,
    )

    return {
        "goals": list(plan_view.training_goals),
        "coverage": {axis: status.to_dict() for axis, status in coverage.items()},
        "top_completed_runs": [record.to_dict() for record in registry_summary.top_completed_runs],
        "queue": [item.to_dict() for item in queue],
        "active_run_study": active_run.study_name if active_run else None,
        "downstream_awareness": plan_view.downstream_awareness,
    }


def _looks_downstream_only_goal(goal: str) -> bool:
    lowered = goal.lower()
    return "plot" in lowered or "evaluation" in lowered or "graph" in lowered


def _queue_item_meta(item: QueueItem) -> dict[str, Any]:
    if item.study_name is not None:
        return {"study_name": item.study_name}
    if item.proposal is not None:
        meta: dict[str, Any] = {
            "study_name": item.proposal.new_study_name,
            **item.proposal.changes,
        }
        return meta
    return {}


def _extract_axis_value(meta: Mapping[str, Any], axis: str) -> Any:
    if axis == "phi_variant":
        if meta.get("phi_variant") is not None:
            return meta.get("phi_variant")
        model = meta.get("model")
        if isinstance(model, Mapping) and model.get("variant_name") is not None:
            return model.get("variant_name")
        return None

    if axis == "stage":
        study_name = str(meta.get("study_name", ""))
        return _infer_stage(study_name)

    return meta.get(axis)


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


def _is_blocked_value(axis: str, value: Any, plan_view: TrainingPlanView) -> bool:
    blocked_axes = plan_view.blocked_axes
    currently_available = plan_view.currently_available

    if axis == "phi_variant":
        blocked = blocked_axes.get(str(value), {})
        if isinstance(blocked, Mapping) and blocked.get("blocked") is True:
            return True
        available = currently_available.get("phi_variants")
        if isinstance(available, list) and value not in available:
            return True

    if axis == "pair_variant":
        blocked = blocked_axes.get(str(value), {})
        if isinstance(blocked, Mapping) and blocked.get("blocked") is True:
            return True
        available = currently_available.get("pair_variants")
        if isinstance(available, list) and value not in available:
            return True

    return False


def _value_in_required(value: Any, required: tuple[Any, ...]) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return any(isinstance(req, (int, float)) and math.isclose(float(value), float(req), rel_tol=1e-9, abs_tol=1e-9) for req in required)
    return value in required


def _stable_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if value is None:
            continue
        if value not in out:
            out.append(value)
    return tuple(out)
