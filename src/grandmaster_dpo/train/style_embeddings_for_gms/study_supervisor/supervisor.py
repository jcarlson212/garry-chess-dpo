from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Literal, TypedDict, Optional

import psutil
from typer.cli import state
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from grandmaster_dpo.train.style_embeddings_for_gms.train_configs import STUDIES, make_config


# =========================
# Paths / config
# =========================

SUPERVISOR_DIR = Path("./src/grandmaster_dpo/train/style_embeddings_for_gms/study_supervisor")
QUEUE_PATH = SUPERVISOR_DIR / "queue.json"
RUNS_DIR = SUPERVISOR_DIR / "runs"
PLAN_PATH = SUPERVISOR_DIR / "experiment_plan.json"
REGISTRY_PATH = SUPERVISOR_DIR / "run_registry.jsonl"
STATE_SNAPSHOT_PATH = SUPERVISOR_DIR / "state_snapshot.json"

# File system utility env variables
EXPERIMENT2_DIR = Path("./final_experiments_for_paper/experiment2_style_model")
TRAINING_SUMMARY_DIR = EXPERIMENT2_DIR / "training_summary"
TRAINED_MODELS_DIR = EXPERIMENT2_DIR / "trained_models"
FILESYSTEM_SNAPSHOT_LIMIT = 200
SUMMARY_PREVIEW_EVENT_LIMIT = 20
TRAIN_CONFIGS_PATH = Path("./src/grandmaster_dpo/train/style_embeddings_for_gms/train_configs.py")
MAX_CONFIG_LIBRARY_ITEMS = 200

# Proposing Studies
PROPOSED_STUDIES_JSONL = SUPERVISOR_DIR / "proposed_studies.jsonl"
PROPOSED_SNIPPETS_PY = SUPERVISOR_DIR / "proposed_train_configs_snippets.py"

POLL_SECONDS = 60 # 1 minute
STALL_SECONDS = 15 * 60
SUMMARY_TAIL_LINES = 300
MAX_LLM_QUEUE_ADDS = 2

LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "nvidia/nemotron-3-super")
OPEN_AI_API_KEY = os.environ.get("OPENAI_API_KEY_GARRY_CHESS", None)
EXPENSIVE_MODEL = "gpt-5"      # or whatever top model you want
CHEAP_MODEL = "gpt-4.1-mini"   # fast/cheap

# Optional LangSmith tracing
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_PROJECT", "style-embeddings-for-chess-gms-study-supervisor")


# =========================
# Types
# =========================

class QueueItem(TypedDict, total=False):
    study_name: str
    priority: int
    created_at: float
    source: str
    reason: str


class RunInfo(TypedDict, total=False):
    study_name: str
    pid: int
    log_path: str
    summary_path: str
    checkpoint_dir: str
    config: dict[str, Any]
    started_at: float
    last_observed_at: float
    status: str
    reason: str
    queue_source: str


class SummaryDigest(TypedDict, total=False):
    summary_exists: bool
    last_event_type: str | None
    last_event_time: float | None
    latest_step_event: dict[str, Any] | None
    latest_epoch_event: dict[str, Any] | None
    run_start_event: dict[str, Any] | None
    run_end_event: dict[str, Any] | None
    timeout_stop_event: dict[str, Any] | None
    best_eval_loss: float | None
    best_eval_margin_cos: float | None
    latest_train_loss: float | None
    latest_pair_acc: float | None
    latest_samples_per_hour: float | None
    latest_total_eta_hours: float | None
    latest_epoch_eta_hours: float | None
    latest_global_step: int | None
    latest_epoch: int | None
    event_count: int
    summary_tail: str


class Observation(TypedDict, total=False):
    process_alive: bool
    log_exists: bool
    last_log_mtime: float | None
    log_tail: str
    status: Literal["idle", "running", "finished_success", "finished_failed", "stalled", "failed"]
    summary: SummaryDigest
    reason: str | None


class SupervisorState(TypedDict, total=False):
    active_run: RunInfo | None
    queue: list[QueueItem]
    completed: list[dict[str, Any]]
    failed: list[dict[str, Any]]
    latest_observation: Observation | None
    llm_decision: dict[str, Any] | None
    last_action: str | None
    experiment_plan: dict[str, Any]
    registry_summary: dict[str, Any]
    experiment_context: dict[str, Any]
    filesystem_context: dict[str, Any]
    summary_library_context: dict[str, Any]
    config_library_context: dict[str, Any]
    llm_calls: int

# =========================
# File System Helpers
# =========================

def recover_active_run() -> RunInfo | None:
    # 1) try state snapshot
    snapshot = load_json_file(STATE_SNAPSHOT_PATH, {})
    active = snapshot.get("active_run")
    if active and active.get("pid") and process_alive(active["pid"]):
        return active

    # 2) try registry fallback
    events = load_registry_events()
    last_started = None
    terminal = {}

    for ev in events:
        study_name = ev.get("study_name")
        if not study_name:
            continue

        if ev.get("event") == "run_started":
            last_started = ev
        elif ev.get("event") in {"run_finished", "run_failed", "run_killed"}:
            terminal[study_name] = ev["event"]

    if last_started:
        study_name = last_started.get("study_name")
        pid = last_started.get("pid")
        if (
            study_name
            and pid
            and study_name not in terminal
            and process_alive(pid)
            and study_name in STUDIES
        ):
            cfg = STUDIES[study_name]
            return RunInfo(
                study_name=study_name,
                pid=pid,
                log_path=str(RUNS_DIR / study_name / "stdout.log"),
                summary_path=str(cfg.summary_path()),
                checkpoint_dir=str(cfg.checkpoint_dir()),
                config=cfg.to_dict(),
                started_at=last_started.get("time", time.time()),
                last_observed_at=time.time(),
                status="running",
                queue_source=last_started.get("queue_source", "recovered"),
            )

    return None

def build_config_library_context(max_items: int = MAX_CONFIG_LIBRARY_ITEMS) -> dict[str, Any]:
    studies = []
    for i, (study_name, cfg) in enumerate(sorted(STUDIES.items(), key=lambda x: x[0])):
        if i >= max_items:
            break
        meta = config_to_meta(study_name, cfg)
        studies.append(meta)

    by_pair_variant = {}
    by_phi_variant = {}
    by_stage_guess = {
        "debug": [],
        "screen": [],
        "finalist": [],
        "ablation": [],
        "other": [],
    }

    for row in studies:
        pair_variant = row.get("pair_variant")
        phi_variant = row.get("phi_variant")
        study_name = row.get("study_name", "")

        if pair_variant is not None:
            by_pair_variant.setdefault(str(pair_variant), []).append(study_name)
        if phi_variant is not None:
            by_phi_variant.setdefault(str(phi_variant), []).append(study_name)

        lowered = study_name.lower()
        if "debug" in lowered:
            by_stage_guess["debug"].append(study_name)
        elif "screen" in lowered:
            by_stage_guess["screen"].append(study_name)
        elif "final" in lowered:
            by_stage_guess["finalist"].append(study_name)
        elif "ablate" in lowered or "ablation" in lowered:
            by_stage_guess["ablation"].append(study_name)
        else:
            by_stage_guess["other"].append(study_name)

    return {
        "train_configs_path": str(TRAIN_CONFIGS_PATH),
        "num_defined_studies": len(STUDIES),
        "studies": studies,
        "by_pair_variant": by_pair_variant,
        "by_phi_variant": by_phi_variant,
        "by_stage_guess": by_stage_guess,
    }

def safe_list_dir(path: Path, max_entries: int = FILESYSTEM_SNAPSHOT_LIMIT) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "entries": []}

    entries = []
    for i, child in enumerate(sorted(path.iterdir(), key=lambda p: p.name)):
        if i >= max_entries:
            entries.append({"name": "...truncated...", "type": "sentinel"})
            break
        entries.append({
            "name": child.name,
            "type": "dir" if child.is_dir() else "file",
        })
    return {
        "path": str(path),
        "exists": True,
        "entries": entries,
    }


def build_filesystem_context() -> dict[str, Any]:
    return {
        "experiment2_root": safe_list_dir(EXPERIMENT2_DIR, max_entries=50),
        "training_summary": safe_list_dir(TRAINING_SUMMARY_DIR, max_entries=100),
        "trained_models": safe_list_dir(TRAINED_MODELS_DIR, max_entries=100),
    }

# =========================
# Persistence helpers
# =========================

def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return default


def save_json_file(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_queue() -> list[QueueItem]:
    data = load_json_file(QUEUE_PATH, [])
    return sorted(data, key=lambda x: (-int(x.get("priority", 0)), float(x.get("created_at", 0.0))))


def save_queue(queue: list[QueueItem]) -> None:
    save_json_file(QUEUE_PATH, queue)


def load_experiment_plan() -> dict[str, Any]:
    return load_json_file(PLAN_PATH, {
        "experiment_family": "style_embeddings_for_gms",
        "goals": [],
        "required_ablations": {},
        "allowed_axes": {},
        "promotion_rules": {},
    })


def append_registry_event(row: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_registry_events() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# =========================
# Study / config helpers
# =========================

def render_make_config_snippet(new_study_name: str, cfg: Any) -> str:
    return f'''    "{new_study_name}": make_config(
        study_name="{new_study_name}",
        train_dir="{cfg.train_dir}",
        eval_dir="{cfg.eval_dir}",
        pair_variant="{cfg.pair_variant}",
        seed={cfg.seed},
        embedding_dim={cfg.model.embedding_dim},
        batch_size={cfg.batch_size},
        lr={cfg.lr},
        tau={cfg.tau},
        phi_variant="{cfg.model.variant_name}",
        epochs={cfg.epochs},
        max_steps_per_epoch={cfg.max_steps_per_epoch},
        max_eval_batches={cfg.max_eval_batches},
        num_workers={cfg.num_workers},
        max_train_rows={cfg.max_train_rows},
        max_eval_rows={cfg.max_eval_rows},
    ),
'''

def append_proposed_snippet(new_study_name: str) -> None:
    if new_study_name not in STUDIES:
        return
    cfg = STUDIES[new_study_name]
    snippet = render_make_config_snippet(new_study_name, cfg)

    first_write = not PROPOSED_SNIPPETS_PY.exists()
    with PROPOSED_SNIPPETS_PY.open("a", encoding="utf-8") as f:
        if first_write:
            f.write("# Auto-generated study proposals\n\n")
        f.write(snippet)
        f.write("\n")

def append_proposed_study_json(row: dict[str, Any]) -> None:
    PROPOSED_STUDIES_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSED_STUDIES_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def derive_asset_readiness(filesystem_context: dict[str, Any]) -> dict[str, Any]:
    root_entries = {e["name"] for e in filesystem_context.get("experiment2_root", {}).get("entries", []) if e.get("type") != "sentinel"}

    return {
        "pairs_v1_ready": "pairs_v1_cached" in root_entries,
        "pairs_v2_ready": "pairs_v2_cached" in root_entries,
        "pairs_v3_ready": "pairs_v3_cached" in root_entries,
        "training_summary_ready": "training_summary" in root_entries,
        "trained_models_ready": "trained_models" in root_entries,
    }

def list_training_summary_files(max_entries: int = 200) -> list[str]:
    if not TRAINING_SUMMARY_DIR.exists():
        return []
    out = []
    for i, p in enumerate(sorted(TRAINING_SUMMARY_DIR.glob("*.jsonl"), key=lambda x: x.name)):
        if i >= max_entries:
            break
        out.append(p.name)
    return out


def read_training_summary_file(filename: str, max_events: int = SUMMARY_PREVIEW_EVENT_LIMIT) -> dict[str, Any]:
    path = TRAINING_SUMMARY_DIR / filename
    if not path.exists():
        return {"exists": False, "filename": filename}

    rows = read_jsonl_tail(path, max_lines=max_events)
    digest = summarize_events(path)

    return {
        "exists": True,
        "filename": filename,
        "path": str(path),
        "digest": digest,
        "tail_events": rows[-max_events:],
    }


def build_summary_library_context(max_files: int = 50) -> dict[str, Any]:
    files = list_training_summary_files(max_entries=max_files)

    # keep this lightweight: only expose filenames + compact digests
    summaries = []
    for fname in files:
        path = TRAINING_SUMMARY_DIR / fname
        digest = summarize_events(path)
        summaries.append({
            "filename": fname,
            "best_eval_loss": digest.get("best_eval_loss"),
            "best_eval_margin_cos": digest.get("best_eval_margin_cos"),
            "latest_pair_acc": digest.get("latest_pair_acc"),
            "latest_samples_per_hour": digest.get("latest_samples_per_hour"),
            "latest_epoch": digest.get("latest_epoch"),
            "latest_global_step": digest.get("latest_global_step"),
            "last_event_type": digest.get("last_event_type"),
        })

    return {
        "available_summary_files": files,
        "summary_digests": summaries,
    }

def config_to_meta(study_name: str, cfg: Any) -> dict[str, Any]:
    return {
        "study_name": study_name,
        "pair_variant": getattr(cfg, "pair_variant", None),
        "tau": getattr(cfg, "tau", None),
        "lr": getattr(cfg, "lr", None),
        "batch_size": getattr(cfg, "batch_size", None),
        "epochs": getattr(cfg, "epochs", None),
        "max_steps_per_epoch": getattr(cfg, "max_steps_per_epoch", None),
        "max_eval_batches": getattr(cfg, "max_eval_batches", None),
        "max_train_rows": getattr(cfg, "max_train_rows", None),
        "max_eval_rows": getattr(cfg, "max_eval_rows", None),
        "phi_variant": getattr(getattr(cfg, "model", None), "variant_name", None),
        "embedding_dim": getattr(getattr(cfg, "model", None), "embedding_dim", None),
        "train_dir": getattr(cfg, "train_dir", None),
        "eval_dir": getattr(cfg, "eval_dir", None),
    }


def build_runinfo_for_study(study_name: str, queue_source: str = "queue") -> RunInfo:
    cfg = STUDIES[study_name]
    return RunInfo(
        study_name=study_name,
        log_path=str(RUNS_DIR / study_name / "stdout.log"),
        summary_path=str(cfg.summary_path()),
        checkpoint_dir=str(cfg.checkpoint_dir()),
        config=cfg.to_dict(),
        queue_source=queue_source,
    )


def materialize_proposed_study(proposal: dict[str, Any]) -> Optional[str]:
    """
    proposal format:
    {
      "base_study": "screen_v1_phi0_tau0_10",
      "new_study_name": "screen_v1_phi0_tau0_25_neighbor",
      "changes": {"tau": 0.25, "epochs": 3, ...}
    }
    """
    base_study = proposal.get("base_study")
    new_study_name = proposal.get("new_study_name")
    changes = proposal.get("changes", {})

    if not base_study or not new_study_name or base_study not in STUDIES:
        return None
    if new_study_name in STUDIES:
        return new_study_name

    base = STUDIES[base_study]

    try:
        new_cfg = make_config(
            study_name=new_study_name,
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
        STUDIES[new_study_name] = new_cfg
        return new_study_name
    except Exception:
        return None


# =========================
# Process / IO helpers
# =========================

def process_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def tail_text(path: str | Path, max_bytes: int = 20000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("rb") as f:
        size = p.stat().st_size
        f.seek(max(0, size - max_bytes))
        return f.read().decode("utf-8", errors="ignore")


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
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def summarize_events(summary_path: str | Path) -> SummaryDigest:
    rows = read_jsonl_tail(summary_path, max_lines=SUMMARY_TAIL_LINES)
    digest: SummaryDigest = {
        "summary_exists": Path(summary_path).exists(),
        "last_event_type": None,
        "last_event_time": None,
        "latest_step_event": None,
        "latest_epoch_event": None,
        "run_start_event": None,
        "run_end_event": None,
        "timeout_stop_event": None,
        "best_eval_loss": None,
        "best_eval_margin_cos": None,
        "latest_train_loss": None,
        "latest_pair_acc": None,
        "latest_samples_per_hour": None,
        "latest_total_eta_hours": None,
        "latest_epoch_eta_hours": None,
        "latest_global_step": None,
        "latest_epoch": None,
        "event_count": len(rows),
        "summary_tail": "\n".join(json.dumps(r) for r in rows[-10:]),
    }

    for row in rows:
        event = row.get("event")
        digest["last_event_type"] = event
        digest["last_event_time"] = row.get("time", digest["last_event_time"])

        if event == "run_start":
            digest["run_start_event"] = row
        elif event == "run_end":
            digest["run_end_event"] = row
        elif event == "timeout_stop":
            digest["timeout_stop_event"] = row
        elif event == "step_end":
            digest["latest_step_event"] = row
            digest["latest_train_loss"] = row.get("train_loss", digest["latest_train_loss"])
            digest["latest_pair_acc"] = row.get("pair_acc", digest["latest_pair_acc"])
            digest["latest_samples_per_hour"] = row.get("samples_per_hour_inst", digest["latest_samples_per_hour"])
            digest["latest_total_eta_hours"] = row.get("eta_total_hours", digest["latest_total_eta_hours"])
            digest["latest_epoch_eta_hours"] = row.get("eta_epoch_hours", digest["latest_epoch_eta_hours"])
            digest["latest_global_step"] = row.get("global_step", digest["latest_global_step"])
            digest["latest_epoch"] = row.get("epoch", digest["latest_epoch"])
        elif event == "epoch_end":
            digest["latest_epoch_event"] = row
            eval_block = row.get("eval", {})
            train_block = row.get("train", {})
            eval_loss = eval_block.get("loss")
            margin_cos = eval_block.get("margin_cos")
            if isinstance(eval_loss, (int, float)):
                if digest["best_eval_loss"] is None or eval_loss < digest["best_eval_loss"]:
                    digest["best_eval_loss"] = float(eval_loss)
            if isinstance(margin_cos, (int, float)):
                if digest["best_eval_margin_cos"] is None or margin_cos > digest["best_eval_margin_cos"]:
                    digest["best_eval_margin_cos"] = float(margin_cos)
            digest["latest_train_loss"] = train_block.get("loss", digest["latest_train_loss"])
            digest["latest_pair_acc"] = eval_block.get("pair_acc", digest["latest_pair_acc"])
            digest["latest_samples_per_hour"] = row.get("samples_per_hour", digest["latest_samples_per_hour"])
            digest["latest_epoch"] = row.get("epoch", digest["latest_epoch"])

    return digest


def start_run(study_name: str, queue_source: str = "queue") -> RunInfo:
    if study_name not in STUDIES:
        raise ValueError(f"Study not found in STUDIES: {study_name}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / study_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "stdout.log"

    f = open(log_path, "a", buffering=1)
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "src.grandmaster_dpo.train.style_embeddings_for_gms.run_studies",
            "--studies",
            study_name,
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
    )

    cfg = STUDIES[study_name]
    run = RunInfo(
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
    )

    append_registry_event({
        "time": time.time(),
        "event": "run_started",
        "study_name": study_name,
        "pid": proc.pid,
        "queue_source": queue_source,
        "config_meta": config_to_meta(study_name, cfg),
    })
    return run


def stop_run(pid: int, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


# =========================
# Registry / experiment summary
# =========================

def build_registry_summary() -> dict[str, Any]:
    events = load_registry_events()
    by_study: dict[str, dict[str, Any]] = {}

    for ev in events:
        study_name = ev.get("study_name")
        if not study_name:
            continue
        by_study.setdefault(study_name, {
            "study_name": study_name,
            "started": False,
            "finished": False,
            "failed": False,
            "killed": False,
            "best_eval_loss": None,
            "best_eval_margin_cos": None,
            "config_meta": ev.get("config_meta"),
        })

        if ev.get("event") == "run_started":
            by_study[study_name]["started"] = True
        elif ev.get("event") == "run_finished":
            by_study[study_name]["finished"] = True
            if ev.get("best_eval_loss") is not None:
                by_study[study_name]["best_eval_loss"] = ev.get("best_eval_loss")
            if ev.get("best_eval_margin_cos") is not None:
                by_study[study_name]["best_eval_margin_cos"] = ev.get("best_eval_margin_cos")
        elif ev.get("event") == "run_failed":
            by_study[study_name]["failed"] = True
        elif ev.get("event") == "run_killed":
            by_study[study_name]["killed"] = True

    completed = [v for v in by_study.values() if v["finished"]]
    completed_sorted = sorted(
        completed,
        key=lambda x: (
            x["best_eval_loss"] if x["best_eval_loss"] is not None else float("inf"),
            -(x["best_eval_margin_cos"] if x["best_eval_margin_cos"] is not None else float("-inf")),
        ),
    )

    return {
        "num_registry_events": len(events),
        "studies_seen": len(by_study),
        "completed_runs": completed_sorted,
        "failed_or_killed_runs": [v for v in by_study.values() if v["failed"] or v["killed"]],
        "top_completed_runs": completed_sorted[:5],
    }


def build_experiment_context(
    plan: dict[str, Any],
    registry_summary: dict[str, Any],
    queue: list[QueueItem],
    active_run: RunInfo | None,
) -> dict[str, Any]:
    completed = registry_summary.get("completed_runs", [])
    required = plan.get("required_ablations", {})

    coverage = {}
    for axis, required_vals in required.items():
        seen = set()
        for row in completed:
            meta = row.get("config_meta") or {}
            value = meta.get(axis)
            if value is not None:
                seen.add(str(value))
        coverage[axis] = {
            "required": [str(x) for x in required_vals],
            "completed": sorted(seen),
            "remaining": [str(x) for x in required_vals if str(x) not in seen],
        }

    return {
        "goals": plan.get("goals", []),
        "coverage": coverage,
        "top_completed_runs": registry_summary.get("top_completed_runs", []),
        "queue": queue,
        "active_run_study": active_run.get("study_name") if active_run else None,
    }


# =========================
# LangGraph nodes
# =========================

def refresh_context(state: SupervisorState) -> dict[str, Any]:
    plan = load_experiment_plan()
    registry_summary = build_registry_summary()
    queue = state.get("queue", load_queue())
    active_run = state.get("active_run")
    experiment_context = build_experiment_context(plan, registry_summary, queue, active_run)
    filesystem_context = build_filesystem_context()
    summary_library_context = build_summary_library_context()
    config_library_context = build_config_library_context()

    return {
        "experiment_plan": plan,
        "registry_summary": registry_summary,
        "experiment_context": experiment_context,
        "filesystem_context": filesystem_context,
        "summary_library_context": summary_library_context,
        "config_library_context": config_library_context,
    }

def observe_run(state: SupervisorState) -> dict[str, Any]:
    active = state.get("active_run")
    if not active:
        return {
            "latest_observation": Observation(
                process_alive=False,
                log_exists=False,
                last_log_mtime=None,
                log_tail="",
                status="idle",
                summary=SummaryDigest(
                    summary_exists=False,
                    event_count=0,
                    summary_tail="",
                ),
                reason="no_active_run",
            )
        }

    pid = active["pid"]
    log_path = active["log_path"]
    summary_path = active["summary_path"]

    alive = process_alive(pid)
    log_p = Path(log_path)
    log_exists = log_p.exists()
    log_tail = tail_text(log_path)
    last_log_mtime = log_p.stat().st_mtime if log_exists else None

    summary_digest = summarize_events(summary_path)
    now = time.time()

    log_lower = log_tail.lower()
    has_traceback = "traceback" in log_lower or "error:" in log_lower
    has_nan = "nan" in log_lower or " inf" in log_lower

    if alive:
        latest_summary_time = summary_digest.get("last_event_time")
        if latest_summary_time is not None and now - latest_summary_time > STALL_SECONDS:
            status = "stalled"
            reason = "summary_not_advancing"
        elif log_exists and last_log_mtime is not None and now - last_log_mtime > STALL_SECONDS:
            status = "stalled"
            reason = "log_not_advancing"
        elif has_traceback or has_nan:
            status = "failed"
            reason = "traceback_or_nan_in_log"
        else:
            status = "running"
            reason = "healthy"
    else:
        if summary_digest.get("run_end_event") is not None:
            status = "finished_success"
            reason = "run_end_seen"
        elif has_traceback or has_nan:
            status = "finished_failed"
            reason = "dead_process_with_error_log"
        else:
            status = "finished_failed"
            reason = "dead_process_without_run_end"

    return {
        "latest_observation": Observation(
            process_alive=alive,
            log_exists=log_exists,
            last_log_mtime=last_log_mtime,
            log_tail=log_tail[-4000:],
            status=status,
            summary=summary_digest,
            reason=reason,
        )
    }


def deterministic_policy(state: SupervisorState) -> dict[str, Any]:
    obs = state.get("latest_observation") or {}
    queue = state.get("queue", [])
    status = obs.get("status")

    if status == "idle":
        if queue:
            return {
                "llm_decision": {
                    "action": "start_next",
                    "reason": "no active run but queue is non-empty",
                }
            }
        # let llm_plan decide how to bootstrap work
        return {"llm_decision": None}

    if status == "stalled":
        return {"llm_decision": {"action": "kill_run", "reason": obs.get("reason", "stalled")}}

    if status in {"failed", "finished_failed"}:
        return {"llm_decision": {"action": "kill_run", "reason": obs.get("reason", "run failed")}}

    if status == "finished_success":
        return {"llm_decision": {"action": "start_next" if queue else "wait", "reason": "run finished successfully"}}

    return {"llm_decision": None}


def llm_plan(state: SupervisorState) -> dict[str, Any]:
    if state.get("llm_decision") is not None:
        return {}

    client = OpenAI(
        api_key=OPEN_AI_API_KEY
    )

    active = state.get("active_run")
    obs = state.get("latest_observation")
    queue = state.get("queue", [])
    context = state.get("experiment_context", {})
    top_completed = context.get("top_completed_runs", [])
    plan = state.get("experiment_plan", {})
    filesystem_context = state.get("filesystem_context", {})
    summary_library_context = state.get("summary_library_context", {})
    last_inspected_summary = summary_library_context.get("last_inspected_summary")
    asset_readiness = derive_asset_readiness(filesystem_context)  # side effect to enrich filesystem context with asset readiness flags
    config_library_context = state.get("config_library_context", {})
    llm_calls = state.get("llm_calls", 0)
    model = EXPENSIVE_MODEL if llm_calls == 0 else CHEAP_MODEL


    prompt = f"""
You are scheduling ML experiments for chess style embeddings.

Goal:
Maximize useful paper evidence per hour while completing the experiment plan.

Allowed actions:
- continue
- enqueue_configs
- propose_configs
- dequeue_configs
- reprioritize_queue
- inspect_summary_file
- inspect_config
- wait

Constraints:
- Never kill a healthy running job from this node.
- Add at most {MAX_LLM_QUEUE_ADDS} new configs.
- Prefer small, local mutations over broad sweeps.
- Prefer completing required ablations and missing coverage.
- Only return strict JSON. No markdown fences.
- Prefer suggestions that are consistent with actual filesystem readiness. 

IMPORTANT:
If a study_name is not already in the config library, you MUST use "proposal" instead of "study_name".

Current train config library:
{json.dumps(config_library_context, indent=2)}

Filesystem context:
{json.dumps(filesystem_context, indent=2)}

Training summary library context:
{json.dumps(summary_library_context, indent=2)}

Last inspected summary file:
{json.dumps(last_inspected_summary, indent=2)}

Asset readiness:
{json.dumps(asset_readiness, indent=2)}

Current active run:
{json.dumps(active, indent=2)}

Latest observation:
{json.dumps(obs, indent=2)}

Experiment plan:
{json.dumps(plan, indent=2)}

Experiment context:
{json.dumps(context, indent=2)}

Current queue:
{json.dumps(queue, indent=2)}

Top completed runs:
{json.dumps(top_completed, indent=2)}

Valid examples:
{{"action":"continue","reason":"healthy run and ablation coverage still incomplete"}}

{{"action":"enqueue_configs","reason":"neighboring tau study needed","configs":[
  {{"study_name":"screen_v1_phi0_tau0_25","priority":5,"reason":"neighbor_tau"}},
  {{"proposal":{{"base_study":"screen_v1_phi0_tau0_10","new_study_name":"screen_v1_phi0_tau0_25_auto","changes":{{"tau":0.25}}}},"priority":4,"reason":"neighbor_tau"}}
]}}

{{"action":"dequeue_configs","reason":"redundant queue entries","study_names":["some_study_name"]}}

{{"action":"reprioritize_queue","reason":"finalist more important","priorities":[
  {{"study_name":"final_v1_phi1_besttau","priority":10}},
  {{"study_name":"screen_v1_phi0_tau1_25","priority":1}}
]}}

Return only JSON.
"""

    try:
        
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "text"},
        )
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].strip()
        decision = json.loads(content)

    except Exception as e:
        print(f"Error in LLM call: {e}")
        decision = {"action": "wait", "reason": f"llm_error:{type(e).__name__}"}

    allowed = {
        "continue",
        "enqueue_configs",
        "propose_configs",
        "dequeue_configs",
        "reprioritize_queue",
        "inspect_summary_file",
        "inspect_config",
        "wait",
    }
    if decision.get("action") not in allowed:
        decision = {"action": "wait", "reason": f"invalid_action:{decision.get('action')}"}

    return {
        "llm_decision": decision,
        "llm_calls": llm_calls + 1,
    }


def execute_action(state: SupervisorState) -> dict[str, Any]:
    decision = state.get("llm_decision") or {"action": "wait"}
    active = state.get("active_run")
    queue = copy.deepcopy(state.get("queue", []))
    completed = copy.deepcopy(state.get("completed", []))
    failed = copy.deepcopy(state.get("failed", []))
    obs = state.get("latest_observation") or {}

    action = decision.get("action", "wait")

    if action == "kill_run" and active:
        stop_run(active["pid"])
        active["status"] = "killed"
        active["reason"] = decision.get("reason", "killed by policy")

        append_registry_event({
            "time": time.time(),
            "event": "run_killed",
            "study_name": active["study_name"],
            "reason": active["reason"],
        })

        failed.append(active)
        save_queue(queue)
        return {
            "active_run": None,
            "failed": failed,
            "queue": queue,
            "last_action": f"kill_run:{active['study_name']}",
        }
    
    if action == "inspect_config":
        study_name = decision.get("study_name")
        if study_name and study_name in STUDIES:
            config_library_context = copy.deepcopy(state.get("config_library_context", {}))
            cfg = STUDIES[study_name]
            config_library_context["last_inspected_config"] = {
                "study_name": study_name,
                "config_meta": config_to_meta(study_name, cfg),
                "full_config": cfg.to_dict(),
            }
            return {
                "config_library_context": config_library_context,
                "last_action": f"inspect_config:{study_name}",
            }
        return {"last_action": "inspect_config:none"}
    
    if action == "propose_configs":
        proposals = decision.get("configs", [])
        proposed_names = []

        for item in proposals[:MAX_LLM_QUEUE_ADDS]:
            proposal = item.get("proposal")
            if not proposal:
                continue

            append_proposed_study_json({
                "time": time.time(),
                "proposal": proposal,
                "reason": item.get("reason", decision.get("reason", "llm_proposal")),
                "priority": item.get("priority", 0),
                "materialized": False,
            })

            study_name = materialize_proposed_study(proposal)
            if study_name:
                proposed_names.append(study_name)
                append_proposed_snippet(study_name)

                append_proposed_study_json({
                    "time": time.time(),
                    "proposal": proposal,
                    "study_name": study_name,
                    "reason": item.get("reason", decision.get("reason", "llm_proposal")),
                    "priority": item.get("priority", 0),
                    "materialized": True,
                })

        return {
            "last_action": f"propose_configs:{','.join(proposed_names) if proposed_names else 'none'}"
        }
    
    if action == "inspect_summary_file":
        filename = decision.get("filename")
        if filename:
            inspected = read_training_summary_file(filename, max_events=SUMMARY_PREVIEW_EVENT_LIMIT)

            # stash it into state-visible context by returning an updated summary_library_context
            summary_library_context = copy.deepcopy(state.get("summary_library_context", {}))
            summary_library_context["last_inspected_summary"] = inspected

            return {
                "summary_library_context": summary_library_context,
                "last_action": f"inspect_summary_file:{filename}",
            }
        return {"last_action": "inspect_summary_file:none"}

    if action == "start_next":
        if active and obs.get("status") == "finished_success":
            summary = obs.get("summary", {})
            active["status"] = "finished"
            completed.append(active)
            append_registry_event({
                "time": time.time(),
                "event": "run_finished",
                "study_name": active["study_name"],
                "best_eval_loss": summary.get("best_eval_loss"),
                "best_eval_margin_cos": summary.get("best_eval_margin_cos"),
                "latest_samples_per_hour": summary.get("latest_samples_per_hour"),
            })
            active = None
        elif active and obs.get("status") == "finished_failed":
            active["status"] = "failed"
            active["reason"] = obs.get("reason", "finished_failed")
            failed.append(active)
            append_registry_event({
                "time": time.time(),
                "event": "run_failed",
                "study_name": active["study_name"],
                "reason": active["reason"],
            })
            active = None

        if active is None and queue:
            queue = sorted(queue, key=lambda x: (-int(x.get("priority", 0)), float(x.get("created_at", 0.0))))
            next_cfg = queue.pop(0)
            new_run = start_run(next_cfg["study_name"], queue_source=next_cfg.get("source", "queue"))
            save_queue(queue)
            return {
                "active_run": new_run,
                "queue": queue,
                "completed": completed,
                "failed": failed,
                "last_action": f"start_next:{new_run['study_name']}",
            }

        save_queue(queue)
        return {
            "active_run": active,
            "queue": queue,
            "completed": completed,
            "failed": failed,
            "last_action": "start_next:none",
        }

    if action == "enqueue_configs":
        existing = {q["study_name"] for q in queue}
        new_items = decision.get("configs", [])
        added = []

        for item in new_items[:MAX_LLM_QUEUE_ADDS]:
            study_name = item.get("study_name")

            if study_name and study_name not in STUDIES:
                # skip invalid direct enqueue
                continue
            proposal = item.get("proposal")

            if proposal:
                study_name = materialize_proposed_study(proposal)

            if not study_name:
                continue
            if study_name not in STUDIES:
                continue
            if study_name in existing:
                continue

            qitem: QueueItem = {
                "study_name": study_name,
                "priority": int(item.get("priority", 0)),
                "created_at": time.time(),
                "source": "llm",
                "reason": item.get("reason", decision.get("reason", "llm_enqueue")),
            }
            queue.append(qitem)
            existing.add(study_name)
            added.append(study_name)

            append_registry_event({
                "time": time.time(),
                "event": "queued",
                "study_name": study_name,
                "reason": qitem["reason"],
                "priority": qitem["priority"],
                "source": qitem["source"],
            })

        save_queue(queue)
        return {
            "queue": queue,
            "last_action": f"enqueue_configs:{','.join(added) if added else 'none'}",
        }

    if action == "dequeue_configs":
        remove_names = set(decision.get("study_names", []))
        if remove_names:
            queue = [q for q in queue if q["study_name"] not in remove_names]
            for study_name in remove_names:
                append_registry_event({
                    "time": time.time(),
                    "event": "dequeued",
                    "study_name": study_name,
                    "reason": decision.get("reason", "llm_dequeue"),
                })
        save_queue(queue)
        return {
            "queue": queue,
            "last_action": f"dequeue_configs:{','.join(sorted(remove_names)) if remove_names else 'none'}",
        }

    if action == "reprioritize_queue":
        new_priorities = {
            item["study_name"]: int(item["priority"])
            for item in decision.get("priorities", [])
            if "study_name" in item and "priority" in item
        }
        for q in queue:
            if q["study_name"] in new_priorities:
                q["priority"] = new_priorities[q["study_name"]]
        save_queue(queue)
        return {
            "queue": queue,
            "last_action": "reprioritize_queue",
        }

    save_queue(queue)
    return {"queue": queue, "last_action": "continue" if action == "continue" else "wait"}


# =========================
# Build graph
# =========================

graph_builder = StateGraph(SupervisorState)
graph_builder.add_node("refresh_context", refresh_context)
graph_builder.add_node("observe_run", observe_run)
graph_builder.add_node("deterministic_policy", deterministic_policy)
graph_builder.add_node("llm_plan", llm_plan)
graph_builder.add_node("execute_action", execute_action)

graph_builder.add_edge(START, "refresh_context")
graph_builder.add_edge("refresh_context", "observe_run")
graph_builder.add_edge("observe_run", "deterministic_policy")
graph_builder.add_edge("deterministic_policy", "llm_plan")
graph_builder.add_edge("llm_plan", "execute_action")
graph_builder.add_edge("execute_action", END)

graph = graph_builder.compile()


# =========================
# Main loop
# =========================

def main() -> None:
    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    recovered_active_run = recover_active_run()

    state: SupervisorState = {
        "active_run": recovered_active_run,
        "queue": load_queue(),
        "completed": [],
        "failed": [],
        "latest_observation": None,
        "llm_decision": None,
        "last_action": None,
        "experiment_plan": load_experiment_plan(),
        "registry_summary": build_registry_summary(),
        "experiment_context": {},
        "llm_calls": 0,
    }

    while True:
        try:
            state = graph.invoke(state)
        except Exception as e:
            state["last_action"] = f"graph_error:{type(e).__name__}"
            print(f"[supervisor] graph error: {e}")

        snapshot = {
            "time": time.time(),
            "last_action": state.get("last_action"),
            "active_run": state.get("active_run"),
            "latest_status": (state.get("latest_observation") or {}).get("status"),
            "latest_reason": (state.get("latest_observation") or {}).get("reason"),
            "queue": state.get("queue", []),
            "experiment_context": state.get("experiment_context", {}),
        }
        save_json_file(STATE_SNAPSHOT_PATH, snapshot)

        print(json.dumps({
            "time": snapshot["time"],
            "last_action": snapshot["last_action"],
            "active_run": snapshot["active_run"],
            "latest_status": snapshot["latest_status"],
            "latest_reason": snapshot["latest_reason"],
            "queue_size": len(snapshot["queue"]),
        }, indent=2))

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()