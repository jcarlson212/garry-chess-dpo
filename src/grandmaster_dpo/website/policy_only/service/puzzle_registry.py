from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PuzzleScenario:
    scenario_id: str
    category: str
    fen: str
    phase: str
    difficulty_estimate_elo: str
    difficulty_numeric_elo: int
    length_bin: str
    sampled_rollout_length_plies: int
    root_eval_cp: int
    root_eval_status: str
    light_tree: dict[str, Any]
    trajectory: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def player_color(self) -> str:
        return "white" if self.fen.split()[1] == "w" else "black"


class PuzzleScenarioRegistry:
    def __init__(self, scenarios: list[PuzzleScenario], *, source_path: str) -> None:
        self._scenarios = scenarios
        self.source_path = source_path
        self._by_category: dict[str, list[PuzzleScenario]] = defaultdict(list)
        self._by_id: dict[str, PuzzleScenario] = {}
        for scenario in scenarios:
            self._by_category[scenario.category].append(scenario)
            self._by_id[scenario.scenario_id] = scenario

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PuzzleScenarioRegistry":
        path = Path(path)
        scenarios: list[PuzzleScenario] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                scenarios.append(
                    PuzzleScenario(
                        scenario_id=str(row["scenario_id"]),
                        category=str(row["scenario_type"]),
                        fen=str(row["fen"]),
                        phase=str(row.get("phase") or "unknown"),
                        difficulty_estimate_elo=str(row.get("difficulty_estimate_elo") or "unknown"),
                        difficulty_numeric_elo=_parse_difficulty_estimate_elo(
                            str(row.get("difficulty_estimate_elo") or "unknown")
                        ),
                        length_bin=str(row.get("length_bin") or ""),
                        sampled_rollout_length_plies=int(row.get("sampled_rollout_length_plies") or 0),
                        root_eval_cp=int(row.get("root_eval_cp") or 0),
                        root_eval_status=str(row.get("root_eval_status") or "unknown"),
                        light_tree=row.get("light_tree") or {},
                        trajectory=[
                            item for item in (row.get("trajectory") or []) if isinstance(item, dict)
                        ],
                        raw=row,
                    )
                )
        if not scenarios:
            raise RuntimeError(f"No puzzle scenarios found in {path}")
        return cls(scenarios, source_path=str(path))

    def list_categories(self) -> list[tuple[str, int]]:
        counts = Counter(s.category for s in self._scenarios)
        return sorted(counts.items())

    def random_for_category(self, category: str, *, rng: random.Random | None = None) -> PuzzleScenario:
        matches = self._by_category.get(category) or []
        if not matches:
            raise KeyError(f"Unknown puzzle category: {category}")
        chooser = rng or random.Random()
        return chooser.choice(matches)

    def sample_for_category_and_target_elo(
        self,
        category: str,
        *,
        target_elo: int,
        normal_mean: float,
        normal_std: float,
        min_elo: int,
        max_elo: int,
        rng: random.Random | None = None,
    ) -> PuzzleScenario:
        matches = self._by_category.get(category) or []
        if not matches:
            raise KeyError(f"Unknown puzzle category: {category}")
        chooser = rng or random.Random()
        sampled = chooser.gauss(float(normal_mean), max(1e-6, float(normal_std)))
        clipped = max(int(min_elo), min(int(max_elo), int(round(sampled))))
        requested = max(int(min_elo), min(int(max_elo), int(target_elo)))
        effective_target = int(round((clipped + requested) / 2.0))
        ranked = sorted(
            matches,
            key=lambda scenario: (
                abs(int(scenario.difficulty_numeric_elo) - effective_target),
                abs(int(scenario.difficulty_numeric_elo) - requested),
                scenario.scenario_id,
            ),
        )
        best_distance = abs(int(ranked[0].difficulty_numeric_elo) - effective_target)
        best_matches = [
            scenario
            for scenario in ranked
            if abs(int(scenario.difficulty_numeric_elo) - effective_target) == best_distance
        ]
        return chooser.choice(best_matches)

    def get_by_id(self, scenario_id: str) -> PuzzleScenario:
        scenario = self._by_id.get(scenario_id)
        if scenario is None:
            raise KeyError(f"Unknown puzzle scenario: {scenario_id}")
        return scenario


def resolve_puzzle_scenarios_path() -> str:
    explicit = os.environ.get("PUZZLE_SCENARIOS_PATH")
    if explicit:
        return explicit
    candidates = [
        "/opt/puzzles/scenario_miner_output.jsonl",
        "./website_data_mining_experiments/scenario_miner_output.jsonl",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "Could not find puzzle scenarios file; set PUZZLE_SCENARIOS_PATH or bundle /opt/puzzles/scenario_miner_output.jsonl"
    )


def resolve_puzzle_scenario_registry() -> PuzzleScenarioRegistry:
    return PuzzleScenarioRegistry.from_jsonl(resolve_puzzle_scenarios_path())


def _parse_difficulty_estimate_elo(raw: str) -> int:
    text = (raw or "").strip()
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
    if match:
        lo = int(match.group(1))
        hi = int(match.group(2))
        return int(round((lo + hi) / 2.0))
    match = re.fullmatch(r"(\d+)\+", text)
    if match:
        lo = int(match.group(1))
        return lo + 100
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return 2000
