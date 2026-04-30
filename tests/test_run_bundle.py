import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _write_html(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html><body>ok</body></html>", encoding="utf-8")
    return path


def test_load_run_bundle_reads_standardized_manifest(tmp_path):
    from synthetic_population_qc.core import RawDataRoots, WorkflowSettings
    from synthetic_population_qc.runs import bundle_table_inventory, load_run_bundle
    from synthetic_population_qc.runs.bundle import (
        ExplorationArtifacts,
        PlanningArtifacts,
        ProcessedArtifacts,
        SyntheticPopulationRun,
        SynthesisArtifacts,
        ValidationArtifacts,
        write_bundle_manifest,
    )

    root = tmp_path / "run"
    for name in ["manifests", "processed", "synthesis", "validation", "exploration"]:
        (root / name).mkdir(parents=True, exist_ok=True)

    person_seed = root / "processed" / "person_seed.parquet"
    household_seed = root / "processed" / "household_seed.parquet"
    people = root / "synthesis" / "people.parquet"
    households = root / "synthesis" / "households.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(person_seed, index=False)
    pd.DataFrame({"x": [1]}).to_parquet(household_seed, index=False)
    pd.DataFrame({"x": [1]}).to_parquet(people, index=False)
    pd.DataFrame({"x": [1]}).to_parquet(households, index=False)
    for rel in [
        "manifests/metadata.json",
        "manifests/workflow_plan.json",
        "manifests/da_coverage.csv",
        "manifests/support_classification.csv",
        "processed/input_contract.csv",
        "processed/preprocessing_audit.csv",
        "processed/person_seed_summary.csv",
        "processed/household_seed_summary.csv",
        "processed/context_manifest.csv",
        "validation/summary_metrics.csv",
        "validation/detail_metrics.csv",
        "validation/support_classification.csv",
        "validation/sparse_handling_report.csv",
        "validation/assignment_route_decisions.csv",
        "validation/household_coherence_audit.csv",
        "exploration/exploration_manifest.json",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}", encoding="utf-8")
        else:
            pd.DataFrame({"x": [1]}).to_csv(path, index=False)
    context_dir = root / "processed" / "context"
    context_dir.mkdir(exist_ok=True)
    pd.DataFrame({"da_code": ["1"]}).to_parquet(context_dir / "housing.parquet", index=False)

    run = SyntheticPopulationRun(
        root=root,
        metadata_json=root / "manifests" / "metadata.json",
        manifest_json=root / "manifests" / "bundle_manifest.json",
        processed=ProcessedArtifacts(
            input_contract_csv=root / "processed" / "input_contract.csv",
            preprocessing_audit_csv=root / "processed" / "preprocessing_audit.csv",
            person_seed_parquet=person_seed,
            household_seed_parquet=household_seed,
            chs_household_seed_parquet=None,
            person_seed_summary_csv=root / "processed" / "person_seed_summary.csv",
            household_seed_summary_csv=root / "processed" / "household_seed_summary.csv",
            context_dir=context_dir,
            context_manifest_csv=root / "processed" / "context_manifest.csv",
        ),
        planning=PlanningArtifacts(
            workflow_plan_json=root / "manifests" / "workflow_plan.json",
            da_coverage_csv=root / "manifests" / "da_coverage.csv",
            support_classification_csv=root / "manifests" / "support_classification.csv",
        ),
        synthesis=SynthesisArtifacts(people_parquet=people, households_parquet=households),
        validation=ValidationArtifacts(
            summary_metrics_csv=root / "validation" / "summary_metrics.csv",
            detail_metrics_csv=root / "validation" / "detail_metrics.csv",
            support_classification_csv=root / "validation" / "support_classification.csv",
            sparse_handling_csv=root / "validation" / "sparse_handling_report.csv",
            assignment_route_decisions_csv=root / "validation" / "assignment_route_decisions.csv",
            household_coherence_audit_csv=root / "validation" / "household_coherence_audit.csv",
        ),
        exploration=ExplorationArtifacts(
            manifest_json=root / "exploration" / "exploration_manifest.json",
            metric_plot_html=_write_html(root / "exploration" / "attribute_tvd.html"),
            support_plot_html=_write_html(root / "exploration" / "support_assessment.html"),
            sparse_plot_html=_write_html(root / "exploration" / "sparse_handling.html"),
            assignment_route_plot_html=_write_html(root / "exploration" / "assignment_route_decisions.html"),
            coherence_plot_html=_write_html(root / "exploration" / "household_coherence.html"),
            dwelling_type_by_household_size_html=_write_html(root / "exploration" / "dwelling_type_by_household_size.html"),
            tenure_by_household_type_html=_write_html(root / "exploration" / "tenure_by_household_type.html"),
            period_built_by_dwelling_type_html=_write_html(root / "exploration" / "period_built_by_dwelling_type.html"),
            commute_mode_by_age_labour_html=_write_html(root / "exploration" / "commute_mode_by_age_labour.html"),
            households_map_html=None,
            owners_share_map_html=None,
        ),
    )
    write_bundle_manifest(run)

    manifest = load_run_bundle(root)
    inventory = bundle_table_inventory(root)

    assert manifest["processed"]["person_seed_parquet"].endswith("person_seed.parquet")
    assert any(path.replace("\\", "/") == "validation/detail_metrics.csv" for path in inventory["relative_path"].tolist())


def test_build_exploration_artifacts_writes_reusable_outputs(tmp_path):
    from synthetic_population_qc.explore.workflow import build_exploration_artifacts

    people = pd.DataFrame(
        {
            "area": ["1", "1", "2"],
            "labour_force_status": [1867, 1867, 9999],
            "age_group": [1, 2, 3],
            "commute_mode": ["public_transit", "private_vehicle", None],
        }
    )
    households = pd.DataFrame(
        {
            "area": ["1", "1", "2"],
            "household_id": ["a", "b", "c"],
            "household_size": ["2", "3", "1"],
            "household_type": ["couple_without_children", "couple_with_children", "one_person"],
            "dwelling_type": ["apartment", "single_detached_house", "apartment"],
            "tenure": ["owner", "renter_or_band", "owner"],
            "period_built": ["2001_to_2005", "2011_to_2015", "1991_to_2000"],
        }
    )
    summary = pd.DataFrame({"unit": ["household"], "attribute": ["tenure"], "tvd": [0.1], "mae_pp": [2.0], "max_abs_pp": [5.0]})
    support = pd.DataFrame(
        {
            "attribute": ["tenure"],
            "unit": ["household"],
            "support_class": ["stable"],
            "min_conditional_weight": [100.0],
            "min_category_weight": [150.0],
            "assignment_route": ["direct_household_assignment"],
        }
    )
    sparse = pd.DataFrame({"attribute": ["commute_mode"], "fallback_rank": [1], "area": ["1"], "unit": ["person"], "used_global_fallback": [False]})
    route = pd.DataFrame(
        {
            "attribute": ["commute_mode"],
            "selected_route": ["sparse_fallback"],
            "planned_route": ["direct_person_assignment"],
            "area": ["1"],
            "unit": ["person"],
            "downgraded_to_sparse": [True],
        }
    )
    coherence = pd.DataFrame({"coherence_issue": ["couple_without_children_requires_size_2"], "area": ["1"], "household_id": ["x"]})

    people_path = tmp_path / "people.parquet"
    households_path = tmp_path / "households.parquet"
    summary_path = tmp_path / "summary.csv"
    support_path = tmp_path / "support.csv"
    sparse_path = tmp_path / "sparse.csv"
    route_path = tmp_path / "routes.csv"
    coherence_path = tmp_path / "coherence.csv"
    people.to_parquet(people_path, index=False)
    households.to_parquet(households_path, index=False)
    summary.to_csv(summary_path, index=False)
    support.to_csv(support_path, index=False)
    sparse.to_csv(sparse_path, index=False)
    route.to_csv(route_path, index=False)
    coherence.to_csv(coherence_path, index=False)

    artifacts = build_exploration_artifacts(
        people_path=people_path,
        households_path=households_path,
        summary_path=summary_path,
        support_path=support_path,
        sparse_path=sparse_path,
        route_path=route_path,
        coherence_path=coherence_path,
        output_dir=tmp_path / "exploration",
        geography_scope="montreal",
        geometry_root=tmp_path / "missing-geometry",
    )

    manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))

    assert artifacts.metric_plot_html.exists()
    assert artifacts.tenure_by_household_type_html.exists()
    assert manifest["households_map_html"] is None


def test_build_exploration_artifacts_tolerates_empty_validation_csvs(tmp_path):
    from synthetic_population_qc.explore.workflow import build_exploration_artifacts

    people_path = tmp_path / "people.parquet"
    households_path = tmp_path / "households.parquet"
    summary_path = tmp_path / "summary.csv"
    support_path = tmp_path / "support.csv"
    sparse_path = tmp_path / "sparse.csv"
    route_path = tmp_path / "routes.csv"
    coherence_path = tmp_path / "coherence.csv"

    pd.DataFrame(
        {
            "area": ["1"],
            "labour_force_status": [1867],
            "age_group": [2],
            "commute_mode": ["public_transit"],
        }
    ).to_parquet(people_path, index=False)
    pd.DataFrame(
        {
            "area": ["1"],
            "household_id": ["h1"],
            "household_size": ["1"],
            "household_type": ["one_person"],
            "dwelling_type": ["apartment"],
            "tenure": ["renter_or_band"],
            "period_built": ["2001_to_2005"],
        }
    ).to_parquet(households_path, index=False)
    summary_path.write_text("", encoding="utf-8")
    support_path.write_text("", encoding="utf-8")
    sparse_path.write_text("", encoding="utf-8")
    route_path.write_text("", encoding="utf-8")
    coherence_path.write_text("", encoding="utf-8")

    artifacts = build_exploration_artifacts(
        people_path=people_path,
        households_path=households_path,
        summary_path=summary_path,
        support_path=support_path,
        sparse_path=sparse_path,
        route_path=route_path,
        coherence_path=coherence_path,
        output_dir=tmp_path / "exploration",
        geography_scope="montreal",
        geometry_root=tmp_path / "missing-geometry",
    )

    assert artifacts.metric_plot_html.exists()
    assert artifacts.support_plot_html.exists()
    assert artifacts.manifest_json.exists()


def test_ensure_run_bundle_falls_back_when_existing_run_is_locked(monkeypatch, tmp_path):
    from synthetic_population_qc.runs.bundle import ensure_run_bundle

    locked_root = tmp_path / "locked_run"
    locked_root.mkdir(parents=True, exist_ok=True)

    def _raise_permission_error(path, onerror=None):
        raise PermissionError("locked")

    monkeypatch.setattr("synthetic_population_qc.runs.bundle.shutil.rmtree", _raise_permission_error)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = ensure_run_bundle(locked_root, overwrite=True)

    assert bundle.root != locked_root
    assert bundle.root.name.startswith("locked_run__rerun_")
    assert bundle.manifests_dir.exists()
    assert any("Could not remove locked run directory" in str(w.message) for w in caught)
