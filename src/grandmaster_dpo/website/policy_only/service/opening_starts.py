from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import chess


ActiveColor = Literal["white", "black"]


@dataclass(frozen=True)
class OpeningStart:
    normalized_name: str
    fen: str
    active_color: ActiveColor


class OpeningStartManager:
    _ALIASES: dict[str, str] = {
        "the_ruy_lopez": "ruy_lopez",
        "sicilian": "sicilian",
        "sicilian_defense": "sicilian",
        "the_sicilian_defense": "sicilian",
        "french": "french",
        "french_defense": "french",
        "the_french_defense": "french",
        "ruy_lopez": "ruy_lopez",
        "spanish": "ruy_lopez",
        "spanish_game": "ruy_lopez",
        "italian": "italian",
        "italian_game": "italian",
        "the_italian_game": "italian",
        "caro_kann": "caro_kann",
        "caro_kann_defense": "caro_kann",
        "the_caro_kann_defense": "caro_kann",
        "queens_gambit": "queens_gambit",
        "queen_s_gambit": "queens_gambit",
        "queen_gambit": "queens_gambit",
        "the_queen_s_gambit": "queens_gambit",
        "the_queens_gambit": "queens_gambit",
        "london": "london",
        "london_system": "london",
        "the_london_system": "london",
        "grunfeld": "grunfeld",
        "grunfeld_defense": "grunfeld",
        "the_grunfeld_defense": "grunfeld",
        "kings_indian": "kings_indian",
        "king_s_indian": "kings_indian",
        "kings_indian_defense": "kings_indian",
        "king_s_indian_defense": "kings_indian",
        "the_kings_indian_defense": "kings_indian",
        "the_king_s_indian_defense": "kings_indian",
        "nimzo_indian": "nimzo_indian",
        "nimzo_indian_defense": "nimzo_indian",
        "the_nimzo_indian_defense": "nimzo_indian",
        "modern": "modern",
        "modern_defense": "modern",
        "the_modern": "modern",
        "the_modern_defense": "modern",
        "english": "english",
        "english_opening": "english",
        "the_english_opening": "english",
        "catalan": "catalan",
        "catalan_opening": "catalan",
        "the_catalan_opening": "catalan",
        "reti": "reti",
        "reti_opening": "reti",
        "the_reti_opening": "reti",
        "scotch": "scotch",
        "scotch_game": "scotch",
        "the_scotch_game": "scotch",
        "kings_gambit": "kings_gambit",
        "king_s_gambit": "kings_gambit",
        "the_kings_gambit": "kings_gambit",
        "the_king_s_gambit": "kings_gambit",
        "pirc": "pirc",
        "pirc_defense": "pirc",
        "the_pirc_defense": "pirc",
        "scandinavian": "scandinavian",
        "scandinavian_defense": "scandinavian",
        "the_scandinavian_defense": "scandinavian",
        "slav": "slav",
        "slav_defense": "slav",
        "the_slav_defense": "slav",
        "semi_slav": "semi_slav",
        "semi_slav_defense": "semi_slav",
        "the_semi_slav_defense": "semi_slav",
        "dutch": "dutch",
        "dutch_defense": "dutch",
        "the_dutch_defense": "dutch",
        "benoni": "benoni",
        "benoni_defense": "benoni",
        "the_benoni_defense": "benoni",
        "queens_indian": "queens_indian",
        "queen_s_indian": "queens_indian",
        "queens_indian_defense": "queens_indian",
        "queen_s_indian_defense": "queens_indian",
        "the_queens_indian_defense": "queens_indian",
        "the_queen_s_indian_defense": "queens_indian",
    }

    _OPENINGS: dict[str, str] = {
        "sicilian": "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "french": "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "ruy_lopez": "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "italian": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "caro_kann": "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "queens_gambit": "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        "london": "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
        "grunfeld": "rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "kings_indian": "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        "nimzo_indian": "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
        "modern": "rnbqk1nr/ppppppbp/6p1/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 1 3",
        "english": "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
        "catalan": "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq - 0 3",
        "reti": "rnbqkbnr/ppp1pppp/8/3p4/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 2",
        "scotch": "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
        "kings_gambit": "rnbqkbnr/pppp1ppp/8/4p3/4PP2/8/PPPP2PP/RNBQKBNR b KQkq - 0 2",
        "pirc": "rnbqkb1r/ppp1pp1p/3p1np1/8/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 4",
        "scandinavian": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "slav": "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
        "semi_slav": "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
        "dutch": "rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
        "benoni": "rnbqkb1r/pp1p1ppp/4pn2/2pP4/2P5/8/PP2PPPP/RNBQKBNR w KQkq - 0 4",
        "queens_indian": "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
    }

    @classmethod
    def normalize_name(cls, name: str) -> str:
        ascii_name = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r"[^a-z0-9]+", "_", ascii_name.strip().lower()).strip("_")
        return cls._ALIASES.get(normalized, normalized)

    def get(self, family_name: str) -> OpeningStart:
        normalized = self.normalize_name(family_name)
        fen = self._OPENINGS.get(normalized)
        if fen is None:
            known = ", ".join(sorted(self._OPENINGS))
            raise KeyError(f"Unknown opening family {family_name!r}; known openings: {known}")
        board = chess.Board(fen)
        active_color: ActiveColor = "white" if board.turn == chess.WHITE else "black"
        return OpeningStart(normalized_name=normalized, fen=board.fen(), active_color=active_color)
