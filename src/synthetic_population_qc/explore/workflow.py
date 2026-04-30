"""Orchestrate reusable exploration artifacts from one completed run bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from synthetic_population_qc.explore.maps import export_map_outputs
from synthetic_population_qc.explore.plots import export_exploration_plots
from synthetic_population_qc.runs.bundle import ExplorationArtifacts


SUMMARY_METRIC_COLUMNS = [
    "unit",
    "attribute",
    "workflow_role",
    "reference_available",
    "reference_source",
    "reference_total",
    "census_available",
    "census_source",
    "census_total",
    "seed_total",
    "seed_to_reference_ratio",
    "seed_to_census_ratio",
    "n_categories",
    "tvd",
    "mae_pp",
    "rmse_pp",
    "max_abs_pp",
]

SUPPORT_COLUMNS = [
    "attribute",
    "unit",
    "workflow_role",
    "assignment_route",
    "support_class",
    "min_conditional_weight",
    "min_category_weight",
]


def _read_csv_artifact(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV artifact, tolerating missing or empty placeholder files."""
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(csv_path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)


def build_exploration_artifacts(
    *,
    people_path: str | Path,
    households_path: str | Path,
    summary_path: str | Path,
    support_path: str | Path,
    sparse_path: str | Path,
    route_path: str | Path,
    coherence_path: str | Path,
    output_dir: str | Path,
    geography_scope: str,
    geometry_root: str | Path | None,
) -> ExplorationArtifacts:
    """Build reusable plots and maps from synthesis and validation outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    people_df = pd.read_parquet(people_path)
    households_df = pd.read_parquet(households_path)
    summary_df = _read_csv_artifact(summary_path, SUMMARY_METRIC_COLUMNS)
    support_df = _read_csv_artifact(support_path, SUPPORT_COLUMNS)
    sparse_df = _read_csv_artifact(sparse_path)
    route_df = _read_csv_artifact(route_path)
    coherence_df = _read_csv_artifact(coherence_path)

    plots = export_exploration_plots(
        summary_df=summary_df,
        support_df=support_df,
        sparse_df=sparse_df,
        route_df=route_df,
        coherence_df=coherence_df,
        people_df=people_df,
        households_df=households_df,
        output_dir=out_dir,
    )
    maps = export_map_outputs(
        people_df=people_df,
        households_df=households_df,
        geography_scope=geography_scope,
        geometry_root=geometry_root,
        output_dir=out_dir,
    )
    manifest_json = out_dir / "exploration_manifest.json"
    manifest_json.write_text(
        json.dumps({key: str(value) if value is not None else None for key, value in {**plots, **maps}.items()}, indent=2),
        encoding="utf-8",
    )

    return ExplorationArtifacts(
        manifest_json=manifest_json,
        metric_plot_html=plots["metric_plot_html"],
        support_plot_html=plots["support_plot_html"],
        sparse_plot_html=plots["sparse_plot_html"],
        assignment_route_plot_html=plots["assignment_route_plot_html"],
        coherence_plot_html=plots["coherence_plot_html"],
        dwelling_type_by_household_size_html=plots["dwelling_type_by_household_size_html"],
        tenure_by_household_type_html=plots["tenure_by_household_type_html"],
        period_built_by_dwelling_type_html=plots["period_built_by_dwelling_type_html"],
        commute_mode_by_age_labour_html=plots["commute_mode_by_age_labour_html"],
        households_map_html=maps["households_map_html"],
        owners_share_map_html=maps["owners_share_map_html"],
    )
