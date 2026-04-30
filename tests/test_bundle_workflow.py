import json
from pathlib import Path

import pandas as pd


def _write_processed_cache(root: Path) -> None:
    seed_dir = root / "seeds"
    context_dir = root / "context"
    seed_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"person_id": [1], "weight": [1.0], "sex": ["male"], "age_group": [1], "labour_force_status": [1867]}).to_parquet(
        seed_dir / "person_seed.parquet",
        index=False,
    )
    pd.DataFrame({"household_id": ["h1"], "weight": [1.0], "household_type": ["one_person"], "household_size": ["1"]}).to_parquet(
        seed_dir / "household_seed_hierarchical.parquet",
        index=False,
    )
    pd.DataFrame({"col": ["ok"]}).to_csv(seed_dir / "input_contract.csv", index=False)
    pd.DataFrame({"col": ["ok"]}).to_csv(seed_dir / "person_seed_summary.csv", index=False)
    pd.DataFrame({"col": ["ok"]}).to_csv(seed_dir / "household_seed_summary.csv", index=False)
    pd.DataFrame({"da_code": ["24660001"]}).to_parquet(context_dir / "housing.parquet", index=False)
    pd.DataFrame(
        [
            {
                "table_name": "housing",
                "rows": 1,
                "columns": 1,
                "has_da_code": True,
                "csv_path": str(context_dir / "housing.csv"),
                "parquet_path": str(context_dir / "housing.parquet"),
            }
        ]
    ).to_csv(root / "context_manifest.csv", index=False)
    pd.DataFrame({"artifact": ["person_seed"], "exists": [True]}).to_csv(root / "preprocessing_audit.csv", index=False)
    pd.DataFrame({"da_code": ["24660001"]}).to_csv(context_dir / "housing.csv", index=False)


def test_build_workflow_input_cache_reuses_existing_artifacts(monkeypatch, tmp_path):
    from synthetic_population_qc.core import RawDataRoots
    from synthetic_population_qc.synth import build_workflow_input_cache

    cache_dir = tmp_path / "prepared"
    _write_processed_cache(cache_dir)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("preprocessing should have been reused")

    monkeypatch.setattr("synthetic_population_qc.ingest.preprocess.export_processed_inputs", _should_not_run)

    artifacts = build_workflow_input_cache(
        raw_data=RawDataRoots.from_paths(
            data_root=tmp_path / "data",
            census_pumf_root=tmp_path / "pumf",
        ),
        cache_dir=cache_dir,
    )

    assert artifacts.person_seed_parquet == cache_dir / "seeds" / "person_seed.parquet"
    assert artifacts.context_manifest_csv == cache_dir / "context_manifest.csv"


def test_build_base_population_cache_passes_explicit_da_codes(monkeypatch, tmp_path):
    from synthetic_population_qc.core import RawDataRoots, WorkflowSettings
    from synthetic_population_qc.synth import build_base_population_cache

    captured = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"area": ["24660001"], "HID": ["1"]}).to_parquet(
            out_dir / "syn_inds_with_hh_core_population.parquet",
            index=False,
        )
        pd.DataFrame({"da_code": ["24660001"]}).to_csv(out_dir / "selected_das_core_population.csv", index=False)
        return pd.DataFrame({"area": ["24660001"]}), {"mode": "generated_for_test"}

    monkeypatch.setattr("synthetic_population_qc.synth.workflow._resolve_or_generate_base_population", _fake_resolve)

    artifacts = build_base_population_cache(
        raw_data=RawDataRoots.from_paths(
            data_root=tmp_path / "data",
            census_pumf_root=tmp_path / "pumf",
        ),
        cache_dir=tmp_path / "base_cache",
        settings=WorkflowSettings.from_paths(
            output_root=tmp_path / "out",
            da_codes=["24660001", "24660002"],
            show_progress=False,
        ),
    )

    metadata = json.loads(artifacts.metadata_json.read_text(encoding="utf-8"))

    assert captured["da_codes"] == ["24660001", "24660002"]
    assert metadata["mode"] == "generated_for_test"
    assert artifacts.base_population_parquet.exists()


def test_run_energy_population_workflow_uses_prebuilt_caches(monkeypatch, tmp_path):
    from synthetic_population_qc.core import RawDataRoots, WorkflowSettings
    from synthetic_population_qc.runs.bundle import PlanningArtifacts
    from synthetic_population_qc.synth.workflow import run_energy_population_workflow

    processed_cache = tmp_path / "prepared"
    _write_processed_cache(processed_cache)
    base_population = tmp_path / "base_cache" / "syn_inds_with_hh_core_population.parquet"
    base_population.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"area": ["24660001"], "HID": ["1"]}).to_parquet(base_population, index=False)

    captured = {}

    def _fake_plan(**kwargs):
        plan_dir = Path(kwargs["output_dir"])
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "workflow_plan.json").write_text("{}", encoding="utf-8")
        pd.DataFrame({"table_name": ["housing"]}).to_csv(plan_dir / "da_coverage.csv", index=False)
        pd.DataFrame({"attribute": ["tenure"]}).to_csv(plan_dir / "support_classification.csv", index=False)
        return PlanningArtifacts(
            workflow_plan_json=plan_dir / "workflow_plan.json",
            da_coverage_csv=plan_dir / "da_coverage.csv",
            support_classification_csv=plan_dir / "support_classification.csv",
        )

    def _fake_native(**kwargs):
        captured.update(kwargs)
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"area": ["24660001"], "person_uid": [1]}).to_parquet(out_dir / "people.parquet", index=False)
        pd.DataFrame({"area": ["24660001"], "household_id": ["h1"]}).to_parquet(out_dir / "households.parquet", index=False)
        pd.DataFrame({"attribute": ["tenure"], "tvd": [0.1]}).to_csv(out_dir / "summary.csv", index=False)
        pd.DataFrame({"attribute": ["tenure"], "category": ["owner"]}).to_csv(out_dir / "details.csv", index=False)
        pd.DataFrame({"attribute": ["tenure"]}).to_csv(out_dir / "support.csv", index=False)
        pd.DataFrame({"attribute": ["tenure"]}).to_csv(out_dir / "sparse.csv", index=False)
        pd.DataFrame({"attribute": ["tenure"]}).to_csv(out_dir / "runtime.csv", index=False)
        pd.DataFrame({"coherence_issue": []}).to_csv(out_dir / "coherence.csv", index=False)

        class _Artifacts:
            people_parquet = out_dir / "people.parquet"
            households_parquet = out_dir / "households.parquet"
            workflow_plan_json = out_dir / "workflow_plan.json"
            assignment_route_csv = out_dir / "runtime.csv"
            household_coherence_csv = out_dir / "coherence.csv"
            summary_csv = out_dir / "summary.csv"
            details_csv = out_dir / "details.csv"
            results_summary_csv = out_dir / "results.csv"
            support_classification_csv = out_dir / "support.csv"
            sparse_handling_csv = out_dir / "sparse.csv"
            metadata_json = out_dir / "metadata.json"

        _Artifacts.workflow_plan_json.write_text("{}", encoding="utf-8")
        _Artifacts.results_summary_csv.write_text("attribute\n", encoding="utf-8")
        _Artifacts.metadata_json.write_text("{}", encoding="utf-8")
        return _Artifacts()

    def _should_not_build_exploration(**kwargs):
        raise AssertionError("exploration should be skipped for cheap batch runs")

    monkeypatch.setattr("synthetic_population_qc.synth.workflow.build_workflow_plan_artifacts", _fake_plan)
    monkeypatch.setattr("synthetic_population_qc.synth.workflow.run_full_energy_aware_workflow", _fake_native)
    monkeypatch.setattr("synthetic_population_qc.synth.workflow.build_exploration_artifacts", _should_not_build_exploration)

    run = run_energy_population_workflow(
        raw_data=RawDataRoots.from_paths(
            data_root=tmp_path / "data",
            census_pumf_root=tmp_path / "pumf",
        ),
        settings=WorkflowSettings.from_paths(
            output_root=tmp_path / "runs",
            run_name="subset_batch",
            da_codes=["24660001"],
            show_progress=False,
            generate_exploration=False,
        ),
        processed_cache=processed_cache,
        base_population_cache=base_population,
    )

    metadata = json.loads(run.metadata_json.read_text(encoding="utf-8"))

    assert captured["processed_inputs_dir"] == processed_cache
    assert captured["base_population_path"] == base_population
    assert captured["da_codes"] == ["24660001"]
    assert run.processed.person_seed_parquet.exists()
    assert metadata["generate_exploration"] is False
