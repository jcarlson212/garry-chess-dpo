from __future__ import annotations

from typing import Any, Dict, List

from .dataset_schema import ExampleRow, PairRow


def is_valid_positive_v1(anchor: ExampleRow, pos: ExampleRow) -> bool:
    return (
        anchor.player_id == pos.player_id
        and anchor.phase == pos.phase
        and anchor.game_type == pos.game_type
    )


def is_valid_negative_v1(anchor: ExampleRow, neg: ExampleRow) -> bool:
    same_game_type = anchor.game_type == neg.game_type
    diff_player = anchor.player_id != neg.player_id
    same_player_diff_phase = (
        anchor.player_id == neg.player_id and anchor.phase != neg.phase
    )
    return same_game_type and (diff_player or same_player_diff_phase)


def is_valid_negative_v2(anchor: ExampleRow, neg: ExampleRow) -> bool:
    return (
        anchor.game_type == neg.game_type
        and anchor.player_id != neg.player_id
    )


def validate_pair_row(pair: Dict[str, Any], variant: str) -> List[str]:
    errors: List[str] = []

    for pos in pair["positives"]:
        if not is_valid_positive_v1(pair["anchor"], pos):
            errors.append(
                f"invalid positive for anchor={pair['anchor']['example_id']}, "
                f"positive={pos['example_id']}"
            )

    if variant == "v1":
        for neg in pair["negatives"]:
            if not is_valid_negative_v1(pair["anchor"], neg):
                errors.append(
                    f"invalid v1 negative for anchor={pair['anchor']['example_id']}, "
                    f"negative={neg['example_id']}"
                )
    elif variant == "v2":
        for neg in pair["negatives"]:
            if not is_valid_negative_v2(pair["anchor"], neg):
                errors.append(
                    f"invalid v2 negative for anchor={pair['anchor']['example_id']}, "
                    f"negative={neg['example_id']}"
                )
    else:
        errors.append(f"unknown variant={variant}")

    return errors