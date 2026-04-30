"""Regression tests for workflow-input discovery and blueprint helpers."""

from pathlib import Path


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_workflow_input_contract_finds_pumf_candidates_and_census_tables(tmp_path):
    from synthetic_population_qc.workflow_inputs import (
        build_workflow_input_contract,
        summarize_workflow_input_contract,
    )

    _touch(tmp_path / "data" / "raw" / "PUMF" / "ind" / "data_donnees_2021_ind_v2.csv")
    _touch(tmp_path / "data" / "raw" / "PUMF" / "heir" / "data_donnees_2021_hier_v2.csv")

    raw_root = tmp_path / "data" / "raw" / "census" / "DA scale"
    for subdir, filename in [
        ("dwelling char", "V818MNIvbF_data.csv"),
        ("housing", "GxvbVaVRu_data.csv"),
        ("commute", "ASmEBfq1h9pC_data.csv"),
        ("imm, citiz", "iilmxSwOn_data.csv"),
        ("income", "tEVGMRS0SSexK_data.csv"),
        ("education", "udDo1DFAvI_data.csv"),
        ("labour", "EVtduVuRucOlsLjw_data.csv"),
        ("hh type, size", "sl5brrkpJ5_data.csv"),
    ]:
        _touch(raw_root / subdir / filename)
        _touch(raw_root / subdir / "New Text Document.txt")

    contract = build_workflow_input_contract(tmp_path)
    summary = summarize_workflow_input_contract(tmp_path)

    assert contract.individual_pumf_candidates
    assert contract.household_pumf_candidates
    assert "housing" in contract.available_census_tables
    assert summary.loc[summary["artifact_name"] == "household_seed_microdata", "found"].iloc[0]


def test_workflow_input_contract_supports_external_census_and_chs_roots(tmp_path):
    from synthetic_population_qc.workflow_inputs import build_workflow_input_contract

    data_root = tmp_path / "urban-energy-data"
    census_root = tmp_path / "census-pumf"
    housing_root = tmp_path / "housing-survey"

    raw_root = data_root / "data" / "raw" / "census" / "DA scale"
    for subdir, filename in [
        ("dwelling char", "V818MNIvbF_data.csv"),
        ("housing", "GxvbVaVRu_data.csv"),
        ("commute", "ASmEBfq1h9pC_data.csv"),
        ("imm, citiz", "iilmxSwOn_data.csv"),
        ("income", "tEVGMRS0SSexK_data.csv"),
        ("education", "udDo1DFAvI_data.csv"),
        ("labour", "EVtduVuRucOlsLjw_data.csv"),
        ("hh type, size", "sl5brrkpJ5_data.csv"),
    ]:
        _touch(raw_root / subdir / filename)
        _touch(raw_root / subdir / "New Text Document.txt")

    _touch(census_root / "ind" / "data_donnees_2021_ind_v2.csv")
    _touch(census_root / "heir" / "data_donnees_2021_hier_v2.csv")
    _touch(housing_root / "Chs2022ecl_pumf.csv")

    contract = build_workflow_input_contract(
        data_root,
        census_pumf_root=census_root,
        housing_survey_root=housing_root,
    )

    assert contract.individual_pumf_candidates
    assert contract.household_pumf_candidates
    assert contract.housing_survey_candidates


def test_workflow_input_contract_finds_pumf_under_repo_style_raw_layout(tmp_path):
    from synthetic_population_qc.workflow_inputs import build_workflow_input_contract

    data_root = tmp_path / "urban-energy-data"
    raw_pumf = data_root / "data" / "raw" / "PUMF"
    raw_root = data_root / "data" / "raw" / "census" / "DA scale"

    for subdir, filename in [
        ("dwelling char", "V818MNIvbF_data.csv"),
        ("housing", "GxvbVaVRu_data.csv"),
        ("commute", "ASmEBfq1h9pC_data.csv"),
        ("imm, citiz", "iilmxSwOn_data.csv"),
        ("income", "tEVGMRS0SSexK_data.csv"),
        ("education", "udDo1DFAvI_data.csv"),
        ("labour", "EVtduVuRucOlsLjw_data.csv"),
        ("hh type, size", "sl5brrkpJ5_data.csv"),
    ]:
        _touch(raw_root / subdir / filename)
        _touch(raw_root / subdir / "New Text Document.txt")

    _touch(raw_pumf / "ind" / "data_donnees_2021_ind_v2.csv")
    _touch(raw_pumf / "heir" / "data_donnees_2021_hier_v2.csv")

    contract = build_workflow_input_contract(data_root)

    assert contract.individual_pumf_candidates
    assert contract.household_pumf_candidates


def test_pumf_candidate_finder_accepts_pumf_root_directly(tmp_path):
    from synthetic_population_qc.workflow_inputs import find_household_pumf_candidates, find_individual_pumf_candidates

    pumf_root = tmp_path / "data" / "raw" / "PUMF"
    _touch(pumf_root / "ind" / "data_donnees_2021_ind_v2.csv")
    _touch(pumf_root / "heir" / "data_donnees_2021_hier_v2.csv")

    individual = find_individual_pumf_candidates(pumf_root)
    household = find_household_pumf_candidates(pumf_root)

    assert individual
    assert household


def test_workflow_blueprints_include_household_stage():
    from synthetic_population_qc.workflow_inputs import (
        build_workflow_attribute_blueprint,
        build_workflow_step_blueprint,
    )

    attributes = build_workflow_attribute_blueprint()
    stages = build_workflow_step_blueprint()

    assert attributes.loc[attributes["attribute"] == "dwelling_type", "assignment_method"].iloc[0] == "household_ipf"
    assert attributes.loc[attributes["attribute"] == "commute_mode", "seed_source"].iloc[0] == "individual_pumf"
    assert attributes.loc[attributes["attribute"] == "commute_mode", "status"].iloc[0] == "implemented_harmonized"
    assert "synthesize_households" in stages["step_name"].tolist()
    assert "validate_outputs" in stages["step_name"].tolist()


def test_workflow_blueprint_can_use_chs_for_household_side():
    from synthetic_population_qc.workflow_inputs import build_workflow_attribute_blueprint

    attributes = build_workflow_attribute_blueprint(use_housing_survey=True)

    assert attributes.loc[attributes["attribute"] == "tenure", "seed_source"].iloc[0] == "hierarchical_pumf_plus_chs"
