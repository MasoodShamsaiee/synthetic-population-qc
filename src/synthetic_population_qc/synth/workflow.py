"""Bundle-first orchestration entry points for the supported public workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from synthetic_population_qc.core.types import RawDataRoots, WorkflowSettings
from synthetic_population_qc.explore.workflow import build_exploration_artifacts
from synthetic_population_qc.ingest.preprocess import (
    build_preprocessed_input_cache,
    export_processed_inputs,
)
from synthetic_population_qc.energy_workflow import (
    _resolve_or_generate_base_population,
    run_full_energy_aware_workflow,
)
from synthetic_population_qc.runs.bundle import (
    BasePopulationArtifacts,
    ExplorationArtifacts,
    PlanningArtifacts,
    ProcessedArtifacts,
    SyntheticPopulationRun,
    SynthesisArtifacts,
    ValidationArtifacts,
    ensure_run_bundle,
    load_processed_artifacts,
    write_bundle_manifest,
)
from synthetic_population_qc.scope_selection import resolve_da_scope_codes
from synthetic_population_qc.synth.planning import build_workflow_plan_artifacts


def _copy(src: Path, dst: Path) -> Path:
    """Copy one artifact into the standardized run bundle layout."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _empty_exploration_artifacts(output_dir: Path) -> ExplorationArtifacts:
    """Create placeholder exploration artifacts when plot export is disabled."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = output_dir / "exploration_manifest.json"
    manifest_json.write_text(json.dumps({"generated": False}, indent=2), encoding="utf-8")
    return ExplorationArtifacts(
        manifest_json=manifest_json,
        metric_plot_html=output_dir / "attribute_tvd.html",
        support_plot_html=output_dir / "support_assessment.html",
        sparse_plot_html=output_dir / "sparse_handling.html",
        assignment_route_plot_html=output_dir / "assignment_route_decisions.html",
        coherence_plot_html=output_dir / "household_coherence.html",
        dwelling_type_by_household_size_html=output_dir / "dwelling_type_by_household_size.html",
        tenure_by_household_type_html=output_dir / "tenure_by_household_type.html",
        period_built_by_dwelling_type_html=output_dir / "period_built_by_dwelling_type.html",
        commute_mode_by_age_labour_html=output_dir / "commute_mode_by_age_labour.html",
        households_map_html=None,
        owners_share_map_html=None,
    )


def _artifact_mapping(value: object) -> dict[str, str]:
    """Serialize dataclass-style artifact containers into string-path metadata."""
    if is_dataclass(value):
        raw = asdict(value)
    else:
        raw = {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }
    return {key: str(item) for key, item in raw.items()}


def _evaluated_attributes(summary_metrics_csv: Path) -> list[str]:
    """Return evaluated attributes from validation metrics, tolerating empty CSVs."""
    if not summary_metrics_csv.exists():
        return []
    try:
        summary_df = pd.read_csv(summary_metrics_csv)
    except EmptyDataError:
        return []
    if "attribute" not in summary_df.columns:
        return []
    return summary_df["attribute"].astype(str).unique().tolist()


def _normalize_processed_artifacts(
    processed: ProcessedArtifacts | str | Path | None,
    *,
    raw_data: RawDataRoots,
    output_dir: Path,
    province: str,
) -> ProcessedArtifacts:
    """Resolve processed inputs from either an object, a path, or raw roots."""
    if processed is None:
        return export_processed_inputs(
            raw_roots=raw_data,
            output_dir=output_dir,
            province=province,
        )
    if isinstance(processed, ProcessedArtifacts):
        return processed
    return load_processed_artifacts(processed)


def _materialize_processed_artifacts(processed: ProcessedArtifacts, output_dir: Path) -> ProcessedArtifacts:
    """Copy processed inputs into the run bundle when they live elsewhere."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if processed.context_dir.parent.resolve() == output_dir.resolve():
        return processed

    seed_dir = output_dir / "seeds"
    context_dir = output_dir / "context"
    seed_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    copied = ProcessedArtifacts(
        input_contract_csv=_copy(processed.input_contract_csv, seed_dir / "input_contract.csv"),
        preprocessing_audit_csv=_copy(processed.preprocessing_audit_csv, output_dir / "preprocessing_audit.csv"),
        person_seed_parquet=_copy(processed.person_seed_parquet, seed_dir / "person_seed.parquet"),
        household_seed_parquet=_copy(
            processed.household_seed_parquet,
            seed_dir / "household_seed_hierarchical.parquet",
        ),
        chs_household_seed_parquet=(
            _copy(processed.chs_household_seed_parquet, seed_dir / "household_seed_chs.parquet")
            if processed.chs_household_seed_parquet is not None
            else None
        ),
        person_seed_summary_csv=_copy(
            processed.person_seed_summary_csv,
            seed_dir / "person_seed_summary.csv",
        ),
        household_seed_summary_csv=_copy(
            processed.household_seed_summary_csv,
            seed_dir / "household_seed_summary.csv",
        ),
        context_dir=context_dir,
        context_manifest_csv=output_dir / "context_manifest.csv",
    )
    for path in sorted(processed.context_dir.glob("*")):
        if path.is_file():
            _copy(path, context_dir / path.name)
    _copy(processed.context_manifest_csv, copied.context_manifest_csv)
    return copied


def build_workflow_input_cache(
    *,
    raw_data: RawDataRoots,
    cache_dir: str | Path,
    province: str = "24",
    overwrite: bool = False,
) -> ProcessedArtifacts:
    """Build and cache harmonized inputs without running full synthesis."""
    return build_preprocessed_input_cache(
        raw_roots=raw_data,
        cache_dir=cache_dir,
        province=province,
        overwrite=overwrite,
    )


def build_base_population_cache(
    *,
    raw_data: RawDataRoots,
    cache_dir: str | Path,
    settings: WorkflowSettings | None = None,
    overwrite: bool = False,
) -> BasePopulationArtifacts:
    """Build or reuse the core base-population cache used by the workflow."""
    settings = settings or WorkflowSettings()
    explicit_da_codes = resolve_da_scope_codes(
        da_codes=list(settings.da_codes) if settings.da_codes is not None else None,
        da_scope_name=settings.da_scope_name,
        da_codes_file=settings.da_codes_file,
    )
    root = Path(cache_dir)
    base_population_parquet = root / "syn_inds_with_hh_core_population.parquet"
    selected_das_csv = root / "selected_das_core_population.csv"
    metadata_json = root / "base_population_metadata.json"

    if overwrite and root.exists():
        shutil.rmtree(root)
    if not overwrite and base_population_parquet.exists() and metadata_json.exists():
        return BasePopulationArtifacts(
            root=root,
            base_population_parquet=base_population_parquet,
            selected_das_csv=selected_das_csv if selected_das_csv.exists() else None,
            metadata_json=metadata_json,
        )

    root.mkdir(parents=True, exist_ok=True)
    base_population_df, metadata = _resolve_or_generate_base_population(
        data_root=raw_data.data_root,
        census_pumf_root=raw_data.census_pumf_root,
        output_dir=root,
        base_population_path=raw_data.base_population_path,
        province=settings.province,
        geography_scope=settings.resolved_geography_scope(),
        random_seed=settings.random_seed,
        max_das=settings.max_das,
        da_codes=explicit_da_codes,
        show_progress=settings.show_progress,
    )
    if not base_population_parquet.exists():
        base_population_df.to_parquet(base_population_parquet, index=False)
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return BasePopulationArtifacts(
        root=root,
        base_population_parquet=base_population_parquet,
        selected_das_csv=selected_das_csv if selected_das_csv.exists() else None,
        metadata_json=metadata_json,
    )


def run_energy_population_workflow(
    *,
    raw_data: RawDataRoots,
    settings: WorkflowSettings | None = None,
    processed_cache: ProcessedArtifacts | str | Path | None = None,
    base_population_cache: BasePopulationArtifacts | str | Path | None = None,
) -> SyntheticPopulationRun:
    """Run the official bundle-first workflow and return typed bundle paths."""
    settings = settings or WorkflowSettings()
    bundle = ensure_run_bundle(settings.output_root / settings.run_name, overwrite=settings.overwrite)

    processed_cache_artifacts = _normalize_processed_artifacts(
        processed_cache,
        raw_data=raw_data,
        output_dir=bundle.processed_dir,
        province=settings.province,
    )
    # Materialize inputs inside the bundle so downstream consumers only need
    # the standardized run root, even when cached inputs were built elsewhere.
    processed = _materialize_processed_artifacts(processed_cache_artifacts, bundle.processed_dir)
    planning: PlanningArtifacts = build_workflow_plan_artifacts(
        raw_roots=raw_data,
        processed=processed,
        output_dir=bundle.manifests_dir,
        method=settings.method,
        province=settings.province,
        geography_scope=settings.resolved_geography_scope(),
        max_das=settings.max_das if settings.da_codes is None else len(settings.da_codes),
        da_codes=list(settings.da_codes) if settings.da_codes is not None else None,
        da_scope_name=settings.da_scope_name,
        da_codes_file=settings.da_codes_file,
        age_group_scheme=settings.age_group_scheme,
        age_group_breaks=list(settings.age_group_breaks) if settings.age_group_breaks is not None else None,
    )

    explicit_da_codes = resolve_da_scope_codes(
        da_codes=list(settings.da_codes) if settings.da_codes is not None else None,
        da_scope_name=settings.da_scope_name,
        da_codes_file=settings.da_codes_file,
    )

    if isinstance(base_population_cache, BasePopulationArtifacts):
        base_population_path = base_population_cache.base_population_parquet
    elif base_population_cache is not None:
        base_population_path = Path(base_population_cache)
    else:
        base_population_path = raw_data.base_population_path

    workflow_output = run_full_energy_aware_workflow(
        data_root=raw_data.data_root,
        census_pumf_root=raw_data.census_pumf_root,
        housing_survey_root=raw_data.housing_survey_root,
        base_population_path=base_population_path,
        output_dir=bundle.cache_dir / "energy_workflow",
        province=settings.province,
        geography_scope=settings.resolved_geography_scope(),
        random_seed=settings.random_seed,
        max_das=settings.max_das,
        da_codes=explicit_da_codes,
        processed_inputs_dir=processed_cache_artifacts.context_dir.parent,
        age_group_scheme=settings.age_group_scheme,
        age_group_breaks=list(settings.age_group_breaks) if settings.age_group_breaks is not None else None,
        show_progress=settings.show_progress,
        method=settings.method,
    )

    synthesis = SynthesisArtifacts(
        people_parquet=_copy(workflow_output.people_parquet, bundle.synthesis_dir / "people.parquet"),
        households_parquet=_copy(workflow_output.households_parquet, bundle.synthesis_dir / "households.parquet"),
    )
    validation = ValidationArtifacts(
        summary_metrics_csv=_copy(workflow_output.summary_csv, bundle.validation_dir / "summary_metrics.csv"),
        detail_metrics_csv=_copy(workflow_output.details_csv, bundle.validation_dir / "detail_metrics.csv"),
        support_classification_csv=_copy(
            workflow_output.support_classification_csv,
            bundle.validation_dir / "support_classification.csv",
        ),
        sparse_handling_csv=_copy(
            workflow_output.sparse_handling_csv,
            bundle.validation_dir / "sparse_handling_report.csv",
        ),
        assignment_route_decisions_csv=_copy(
            workflow_output.assignment_route_csv,
            bundle.validation_dir / "assignment_route_decisions.csv",
        ),
        household_coherence_audit_csv=_copy(
            workflow_output.household_coherence_csv,
            bundle.validation_dir / "household_coherence_audit.csv",
        ),
    )
    if settings.generate_exploration:
        exploration = build_exploration_artifacts(
            people_path=synthesis.people_parquet,
            households_path=synthesis.households_parquet,
            summary_path=validation.summary_metrics_csv,
            support_path=validation.support_classification_csv,
            sparse_path=validation.sparse_handling_csv,
            route_path=validation.assignment_route_decisions_csv,
            coherence_path=validation.household_coherence_audit_csv,
            output_dir=bundle.exploration_dir,
            geography_scope=settings.resolved_geography_scope(),
            geometry_root=raw_data.resolved_geometry_root(),
        )
    else:
        exploration = _empty_exploration_artifacts(bundle.exploration_dir)

    metadata = {
        "workflow_name": "energy_population_workflow",
        "workflow_version": "bundle_v1",
        "workflow_method": settings.method,
        "geography_scope": settings.resolved_geography_scope(),
        "da_scope_name": settings.da_scope_name,
        "province": settings.province,
        "random_seed": settings.random_seed,
        "max_das": settings.max_das,
        "da_codes": explicit_da_codes,
        "da_codes_file": str(settings.da_codes_file) if settings.da_codes_file is not None else None,
        "age_group_scheme": settings.age_group_scheme,
        "age_group_breaks": list(settings.age_group_breaks) if settings.age_group_breaks is not None else None,
        "generate_exploration": settings.generate_exploration,
        "raw_data": {
            "data_root": str(raw_data.data_root),
            "census_pumf_root": str(raw_data.census_pumf_root),
            "housing_survey_root": str(raw_data.housing_survey_root) if raw_data.housing_survey_root else None,
            "geometry_root": str(raw_data.resolved_geometry_root()),
            "base_population_path": str(raw_data.base_population_path) if raw_data.base_population_path else None,
        },
        "processed_cache_dir": str(processed_cache_artifacts.context_dir.parent),
        "base_population_cache_path": str(base_population_path) if base_population_path is not None else None,
        "core_generation_attributes": [
            item["attribute"]
            for item in json.loads(planning.workflow_plan_json.read_text(encoding="utf-8")).get("all_workflow_attributes", [])
            if item.get("workflow_role") == "core_generation"
        ],
        "all_workflow_attributes": json.loads(planning.workflow_plan_json.read_text(encoding="utf-8")).get("all_workflow_attributes", []),
        "evaluated_attributes": _evaluated_attributes(validation.summary_metrics_csv),
        "workflow_artifacts": _artifact_mapping(workflow_output),
    }
    metadata_json = bundle.manifests_dir / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run = SyntheticPopulationRun(
        root=bundle.root,
        metadata_json=metadata_json,
        manifest_json=bundle.manifests_dir / "bundle_manifest.json",
        processed=processed,
        planning=planning,
        synthesis=synthesis,
        validation=validation,
        exploration=exploration,
    )
    write_bundle_manifest(run)
    return run
