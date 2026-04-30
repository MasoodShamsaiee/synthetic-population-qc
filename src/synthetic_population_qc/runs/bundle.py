"""Dataclasses and filesystem helpers for standardized workflow bundles."""

from __future__ import annotations

import json
import os
import shutil
import stat
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunBundlePaths:
    """Directory layout for one persisted workflow run."""
    root: Path
    manifests_dir: Path
    processed_dir: Path
    synthesis_dir: Path
    validation_dir: Path
    exploration_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class ProcessedArtifacts:
    """Pointers to processed seeds and harmonized context tables."""
    input_contract_csv: Path
    preprocessing_audit_csv: Path
    person_seed_parquet: Path
    household_seed_parquet: Path
    chs_household_seed_parquet: Path | None
    person_seed_summary_csv: Path
    household_seed_summary_csv: Path
    context_dir: Path
    context_manifest_csv: Path


@dataclass(frozen=True)
class BasePopulationArtifacts:
    """Pointers to a cached base synthetic population selection."""
    root: Path
    base_population_parquet: Path
    selected_das_csv: Path | None
    metadata_json: Path


@dataclass(frozen=True)
class PlanningArtifacts:
    """Planning-stage bundle outputs used by downstream synthesis."""
    workflow_plan_json: Path
    da_coverage_csv: Path
    support_classification_csv: Path
    mapping_audit_csv: Path | None = None


@dataclass(frozen=True)
class SynthesisArtifacts:
    """Canonical synthetic people and household outputs."""
    people_parquet: Path
    households_parquet: Path


@dataclass(frozen=True)
class ValidationArtifacts:
    """Validation reports emitted by the workflow."""
    summary_metrics_csv: Path
    detail_metrics_csv: Path
    support_classification_csv: Path
    sparse_handling_csv: Path
    assignment_route_decisions_csv: Path
    household_coherence_audit_csv: Path


@dataclass(frozen=True)
class ExplorationArtifacts:
    """HTML plots and optional maps generated from one run bundle."""
    manifest_json: Path
    metric_plot_html: Path
    support_plot_html: Path
    sparse_plot_html: Path
    assignment_route_plot_html: Path
    coherence_plot_html: Path
    dwelling_type_by_household_size_html: Path
    tenure_by_household_type_html: Path
    period_built_by_dwelling_type_html: Path
    commute_mode_by_age_labour_html: Path
    households_map_html: Path | None
    owners_share_map_html: Path | None


@dataclass(frozen=True)
class SyntheticPopulationRun:
    """Full typed handle to one workflow run bundle on disk."""
    root: Path
    metadata_json: Path
    manifest_json: Path
    processed: ProcessedArtifacts
    planning: PlanningArtifacts
    synthesis: SynthesisArtifacts
    validation: ValidationArtifacts
    exploration: ExplorationArtifacts

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize the bundle dataclasses into a JSON-friendly manifest payload."""
        def _serialize(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, "__dataclass_fields__"):
                return {key: _serialize(val) for key, val in asdict(value).items()}
            if isinstance(value, dict):
                return {key: _serialize(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize(val) for val in value]
            return value

        return _serialize(asdict(self))


def ensure_run_bundle(root: str | Path, *, overwrite: bool = True) -> RunBundlePaths:
    """Create or reset the standard bundle directory tree for one run."""
    run_root = Path(root)
    if run_root.exists() and overwrite:
        try:
            shutil.rmtree(run_root, onerror=_handle_remove_readonly)
        except PermissionError:
            fallback_root = _fallback_run_root(run_root)
            warnings.warn(
                (
                    f"Could not remove locked run directory {run_root}. "
                    f"Writing this run to {fallback_root} instead."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            run_root = fallback_root
    run_root.mkdir(parents=True, exist_ok=True)
    bundle = RunBundlePaths(
        root=run_root,
        manifests_dir=run_root / "manifests",
        processed_dir=run_root / "processed",
        synthesis_dir=run_root / "synthesis",
        validation_dir=run_root / "validation",
        exploration_dir=run_root / "exploration",
        cache_dir=run_root / "_cache",
    )
    for path in asdict(bundle).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return bundle


def _handle_remove_readonly(func, path, exc_info) -> None:
    """Retry directory cleanup after clearing readonly bits on Windows."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except PermissionError:
        raise


def _fallback_run_root(run_root: Path) -> Path:
    """Choose a timestamped rerun directory when the requested run root is locked."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root.parent / f"{run_root.name}__rerun_{stamp}"
    idx = 1
    while candidate.exists():
        idx += 1
        candidate = run_root.parent / f"{run_root.name}__rerun_{stamp}_{idx}"
    return candidate


def write_bundle_manifest(run: SyntheticPopulationRun) -> Path:
    """Persist the bundle manifest JSON for one run."""
    run.manifest_json.write_text(json.dumps(run.to_manifest_dict(), indent=2), encoding="utf-8")
    return run.manifest_json


def load_run_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Load the JSON manifest plus top-level path shortcuts for a run bundle."""
    root = Path(bundle_root)
    manifest_path = root / "manifests" / "bundle_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paths"] = {
        "root": str(root),
        "manifests": str(root / "manifests"),
        "processed": str(root / "processed"),
        "synthesis": str(root / "synthesis"),
        "validation": str(root / "validation"),
        "exploration": str(root / "exploration"),
    }
    return manifest


def bundle_table_inventory(bundle_root: str | Path) -> pd.DataFrame:
    """List every persisted file in the standardized bundle layout."""
    root = Path(bundle_root)
    rows: list[dict[str, object]] = []
    for group in ["manifests", "processed", "synthesis", "validation", "exploration"]:
        group_dir = root / group
        for path in sorted(group_dir.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "group": group,
                        "relative_path": str(path.relative_to(root)),
                        "suffix": path.suffix.lower(),
                        "size_bytes": path.stat().st_size,
                    }
                )
    return pd.DataFrame(rows)


def load_processed_artifacts(processed_dir: str | Path) -> ProcessedArtifacts:
    """Load the standardized processed-input artifact bundle."""
    root = Path(processed_dir)
    seed_dir = root / "seeds"
    context_dir = root / "context"
    chs_path = seed_dir / "household_seed_chs.parquet"
    return ProcessedArtifacts(
        input_contract_csv=seed_dir / "input_contract.csv",
        preprocessing_audit_csv=root / "preprocessing_audit.csv",
        person_seed_parquet=seed_dir / "person_seed.parquet",
        household_seed_parquet=seed_dir / "household_seed_hierarchical.parquet",
        chs_household_seed_parquet=chs_path if chs_path.exists() else None,
        person_seed_summary_csv=seed_dir / "person_seed_summary.csv",
        household_seed_summary_csv=seed_dir / "household_seed_summary.csv",
        context_dir=context_dir,
        context_manifest_csv=root / "context_manifest.csv",
    )
