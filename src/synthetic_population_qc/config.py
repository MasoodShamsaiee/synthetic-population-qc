"""Default path helpers for the cleaned synthetic population package."""

from __future__ import annotations

import os
from pathlib import Path


DATA_REPO_NAME = "urban-energy-data"
PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_root(cwd: Path | None = None) -> Path:
    """Resolve the repo root, allowing environment overrides for batch runs."""
    env_root = os.environ.get("SYNTHPOP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    if cwd is not None:
        here = Path(cwd).resolve()
        return here.parent if here.name.lower() == "notebooks" else here
    return PACKAGE_PROJECT_ROOT


def default_data_repo_root(cwd: Path | None = None) -> Path:
    """Return the sibling shared-data repository used by the workflow by default."""
    env_root = os.environ.get("SYNTHPOP_SHARED_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return project_root(cwd).parent / DATA_REPO_NAME


def default_geometry_dir(cwd: Path | None = None) -> Path:
    """Return the default geometry directory for DA map outputs."""
    env_root = os.environ.get("SYNTHPOP_GEOMETRY_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return default_data_repo_root(cwd) / "data" / "raw" / "geometry"


def default_output_dir(cwd: Path | None = None) -> Path:
    """Return the default processed-output root for workflow bundles."""
    env_root = os.environ.get("SYNTHPOP_OUTPUT_DIR", os.environ.get("SYNTHPOP_QC_OUTPUT_DIR"))
    if env_root:
        return Path(env_root).resolve()
    return project_root(cwd) / "data" / "processed" / "synthetic_population"
