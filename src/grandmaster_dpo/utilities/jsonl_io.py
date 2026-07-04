#!/usr/bin/env python3
"""Helpers for reading JSONL datasets that may be zstd-compressed.

Large JSONL datasets in this repo (e.g. experiment2 pairs_v1/pairs_v2) are
stored as ``*.jsonl.zst`` to save disk space. These helpers let readers accept
either plain ``*.jsonl`` or ``*.jsonl.zst`` transparently.
"""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable, List

import pyzstd

JSONL_SUFFIXES = (".jsonl", ".jsonl.zst")


def sorted_jsonl_paths(input_dir: Path) -> List[Path]:
    """All *.jsonl and *.jsonl.zst files in input_dir, sorted by dataset name.

    Sorting strips the .zst suffix so shard order matches the uncompressed
    layout. If both foo.jsonl and foo.jsonl.zst exist, only the plain file is
    returned (it is assumed newer / authoritative).
    """
    paths = {p.name: p for p in input_dir.glob("*.jsonl.zst")}
    for p in input_dir.glob("*.jsonl"):
        paths.pop(p.name + ".zst", None)
        paths[p.name] = p
    return [paths[name] for name in sorted(paths, key=lambda n: n.removesuffix(".zst"))]


def open_jsonl_binary(path: Path) -> BinaryIO:
    """Open a .jsonl or .jsonl.zst file for binary line iteration."""
    if path.name.endswith(".zst"):
        return pyzstd.open(path, "rb")
    return path.open("rb")


def iter_jsonl_lines(input_dir: Path) -> Iterable[bytes]:
    """Yield non-empty raw lines from every jsonl(.zst) file in input_dir."""
    for path in sorted_jsonl_paths(input_dir):
        with open_jsonl_binary(path) as f:
            for line in f:
                if line.strip():
                    yield line
