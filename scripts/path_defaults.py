#!/usr/bin/env python3
import os
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SHARED_ROOT = WORKSPACE_ROOT.parent
LEGACY_ROOT = Path("/Users/sota/code/3d_new")


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _first_existing(candidates: Iterable[Optional[Path]]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def resolve_root() -> Path:
    return _env_path("TRACK_GEN_ROOT") or PROJECT_ROOT


def resolve_default_output_root() -> Path:
    return _env_path("TRACK_GEN_OUTPUT_ROOT") or (resolve_root() / "output")


def resolve_default_source_root() -> Path:
    explicit_source = _env_path("TRACK_GEN_SOURCE_ROOT")
    if explicit_source is not None:
        return explicit_source

    data_root = _env_path("TRACK_GEN_DATA_ROOT")
    existing = _first_existing(
        [
            data_root,
            resolve_root() / "data",
            resolve_root() / "data" / "scenes" / "3d-assets",
            resolve_root() / "data" / "scenes",
            SHARED_ROOT / "data",
            WORKSPACE_ROOT / "data",
            PROJECT_ROOT / "data",
            LEGACY_ROOT / "data",
            LEGACY_ROOT / "data" / "scenes" / "3d-assets",
            LEGACY_ROOT / "data" / "scenes",
        ]
    )
    if existing is not None:
        return existing
    return (resolve_root() / "data").resolve()


ROOT = resolve_root()
DEFAULT_OUTPUT_ROOT = resolve_default_output_root()
DEFAULT_SOURCE_ROOT = resolve_default_source_root()
