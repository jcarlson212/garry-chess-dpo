#!/usr/bin/env python3
"""Helpers for loading numpy cache files that may be zstd-compressed.

Large .npy/.npz caches in this repo (e.g. experiment2 pairs_v*_cached and
eval_outputs) are archived as ``*.npy.zst`` / ``*.npz.zst`` to save disk
space. np.load needs a real uncompressed file for mmap, so these helpers
transparently decompress a ``.zst`` sibling in place (removing the ``.zst``)
the first time the file is needed. Plain uncompressed files are used as-is,
so existing workflows are unaffected.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyzstd


def zst_sibling(path: Path) -> Path:
    return path.with_name(path.name + ".zst")


def cache_file_present(path: Path) -> bool:
    """True if the file exists either uncompressed or as a .zst archive."""
    path = Path(path)
    return path.exists() or zst_sibling(path).exists()


def ensure_decompressed(path: Path) -> Path:
    """Decompress path's .zst sibling in place if the plain file is missing.

    Returns path unchanged when it already exists (or when neither form
    exists — the caller's np.load will then raise the usual error). The
    decompressed file is written atomically and the .zst is removed on
    success, so interrupted runs never leave a truncated cache file behind.
    """
    path = Path(path)
    if path.exists():
        return path
    zst = zst_sibling(path)
    if not zst.exists():
        return path
    tmp = path.with_name(f"{path.name}.tmp-decompress-{os.getpid()}")
    try:
        with pyzstd.open(zst, "rb") as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    try:
        zst.unlink()
    except FileNotFoundError:
        pass  # another process already cleaned it up
    return path


def load_npy(path: Path, mmap_mode: Any = None, allow_pickle: bool = False) -> Any:
    """Drop-in np.load that transparently decompresses .zst-archived files."""
    return np.load(ensure_decompressed(Path(path)), mmap_mode=mmap_mode, allow_pickle=allow_pickle)
