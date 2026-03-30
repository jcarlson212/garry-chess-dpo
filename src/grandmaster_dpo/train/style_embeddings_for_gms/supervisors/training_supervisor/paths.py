from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _guess_repo_root(here: Path) -> Path:
    """Prefer cwd (where the user runs the orchestrator) then walk from this file."""
    for start in (Path.cwd().resolve(), here):
        cur = start
        for _ in range(14):
            if (cur / "pyproject.toml").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    return here.parents[6]


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


@dataclass(frozen=True)
class TrainingSupervisorPaths:
    project_root: Path
    src_root: Path
    package_root: Path
    supervisors_root: Path
    training_supervisor_root: Path
    legacy_study_supervisor_root: Path

    state_dir: Path
    runs_dir: Path

    shared_experiment_plan_path: Path

    queue_path: Path
    run_registry_path: Path
    state_snapshot_path: Path
    proposed_studies_path: Path
    materialized_studies_path: Path
    handoff_events_path: Path

    legacy_queue_path: Path
    legacy_run_registry_path: Path
    legacy_state_snapshot_path: Path
    legacy_proposed_studies_path: Path

    experiment2_dir: Path
    training_summary_dir: Path
    trained_models_dir: Path

    @classmethod
    def discover(cls) -> "TrainingSupervisorPaths":
        here = Path(__file__).resolve()
        training_supervisor_root = here.parent
        supervisors_root = training_supervisor_root.parent
        package_root = supervisors_root.parent
        src_root = here.parents[5]
        project_root = _guess_repo_root(here)

        legacy_study_supervisor_root = package_root / "study_supervisor"

        state_dir = training_supervisor_root / "state"
        runs_dir = training_supervisor_root / "runs"

        experiment2_dir = project_root / "final_experiments_for_paper" / "experiment2_style_model"

        shared_experiment_plan_path = _first_existing(
            supervisors_root / "experiment_plan.json",
            legacy_study_supervisor_root / "experiment_plan.json",
            training_supervisor_root / "experiment_plan.json",
        )

        return cls(
            project_root=project_root,
            src_root=src_root,
            package_root=package_root,
            supervisors_root=supervisors_root,
            training_supervisor_root=training_supervisor_root,
            legacy_study_supervisor_root=legacy_study_supervisor_root,
            state_dir=state_dir,
            runs_dir=runs_dir,
            shared_experiment_plan_path=shared_experiment_plan_path,
            queue_path=state_dir / "queue.json",
            run_registry_path=state_dir / "run_registry.jsonl",
            state_snapshot_path=state_dir / "state_snapshot.json",
            proposed_studies_path=state_dir / "proposed_studies.jsonl",
            materialized_studies_path=state_dir / "materialized_studies.json",
            handoff_events_path=state_dir / "handoff_events.jsonl",
            legacy_queue_path=legacy_study_supervisor_root / "queue.json",
            legacy_run_registry_path=legacy_study_supervisor_root / "run_registry.jsonl",
            legacy_state_snapshot_path=legacy_study_supervisor_root / "state_snapshot.json",
            legacy_proposed_studies_path=legacy_study_supervisor_root / "proposed_studies.jsonl",
            experiment2_dir=experiment2_dir,
            training_summary_dir=experiment2_dir / "training_summary",
            trained_models_dir=experiment2_dir / "trained_models",
        )

    def ensure_dirs(self) -> None:
        self.training_supervisor_root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


PATHS = TrainingSupervisorPaths.discover()
