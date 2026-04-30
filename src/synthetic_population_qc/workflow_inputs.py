"""Inspect available workflow inputs and describe the supported pipeline shape.

This module does two related jobs:
1. search the raw-data roots for the census/CHS artifacts the workflow needs
2. expose small tabular blueprints that explain which attributes and methods the
   public workflow supports
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from synthetic_population_qc.context_tables import DA_TABLE_SPECS, resolve_da_table_artifacts, resolve_pumf_root


def resolve_housing_survey_root(
    housing_survey_root: str | Path | None = None,
) -> Path | None:
    """Resolve the CHS root when one is declared or embedded under the raw tree."""
    if housing_survey_root is None:
        return None
    root = Path(housing_survey_root)
    candidates = (
        root,
        root / "CHS",
        root / "raw" / "CHS",
        root / "data" / "raw" / "CHS",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class WorkflowInputContract:
    """Resolved view of the raw artifacts available to the workflow."""

    data_root: Path
    census_pumf_root: Path | None
    housing_survey_root: Path | None
    individual_pumf_candidates: tuple[Path, ...]
    household_pumf_candidates: tuple[Path, ...]
    housing_survey_candidates: tuple[Path, ...]
    available_census_tables: tuple[str, ...]


EXPECTED_DA_CENSUS_TABLES: dict[str, tuple[str, str]] = {
    name: (f"{subdir}/{csv_name}", f"{subdir}/{metadata_name}")
    for name, (subdir, csv_name, metadata_name) in DA_TABLE_SPECS.items()
}


def _candidate_paths(root: Path, patterns: tuple[str, ...], preferred: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    """Collect files matching known patterns under one normalized root."""
    matches: list[Path] = []
    for path in preferred:
        if path.exists():
            matches.append(path)
    if root.exists():
        for pattern in patterns:
            matches.extend(sorted(root.rglob(pattern)))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in matches:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def find_individual_pumf_candidates(
    data_root: str | Path,
    *,
    census_pumf_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return likely 2021 individual PUMF files under the declared roots."""
    root = resolve_pumf_root(data_root, census_pumf_root=census_pumf_root)
    patterns = (
        "cen_ind_*pumf*.dta",
        "cen_ind_*pumf*.sav",
        "*individual*2021*pumf*.dta",
        "*individual*2021*pumf*.sav",
        "data_donnees_2021_ind_v2.csv",
        "data_donnees_2021_ind.dat",
        "ipumf_2021_final_en*.dct",
    )
    preferred = (
        root / "ind" / "data_donnees_2021_ind_v2.csv",
        root / "ind" / "data_donnees_2021_ind.dat",
    )
    return _candidate_paths(root, patterns, preferred=preferred)


def find_household_pumf_candidates(
    data_root: str | Path,
    *,
    census_pumf_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return likely 2021 hierarchical/household PUMF files under the declared roots."""
    root = resolve_pumf_root(data_root, census_pumf_root=census_pumf_root)
    patterns = (
        "cen_hh_*pumf*.dta",
        "cen_hh_*pumf*.sav",
        "*household*2021*pumf*.dta",
        "*household*2021*pumf*.sav",
        "*hh*2021*pumf*.dta",
        "*hh*2021*pumf*.sav",
        "data_donnees_2021_hier_v2.csv",
        "data_donnees_2021_hier_v2.dat",
        "pumf2021_hierarchical_stata.dct",
    )
    preferred = (
        root / "heir" / "data_donnees_2021_hier_v2.csv",
        root / "heir" / "data_donnees_2021_hier_v2.dat",
    )
    return _candidate_paths(root, patterns, preferred=preferred)


def find_housing_survey_candidates(
    *,
    housing_survey_root: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return likely CHS public-use files when a housing-survey root is available."""
    root = resolve_housing_survey_root(housing_survey_root=housing_survey_root)
    if root is None:
        return tuple()
    patterns = (
        "Chs2022ecl_pumf.csv",
        "Chs2022ecl_pumf_bsw.csv",
        "CHS2022ECL_PUMF.txt",
        "CHS2022ECL_PUMF_BSW.txt",
        "chs2022ecl_pumf.dct",
        "chs2022ecl_pumf_bsw.dct",
    )
    return _candidate_paths(root, patterns)


def build_workflow_input_contract(
    data_root: str | Path,
    *,
    census_pumf_root: str | Path | None = None,
    housing_survey_root: str | Path | None = None,
) -> WorkflowInputContract:
    """Resolve which required raw inputs are present under the configured roots."""
    root = Path(data_root)
    census_root = resolve_pumf_root(root, census_pumf_root=census_pumf_root)
    housing_root = resolve_housing_survey_root(housing_survey_root=housing_survey_root)
    table_artifacts = resolve_da_table_artifacts(root)
    available_tables = tuple(
        sorted(
            name
            for name, (csv_path, meta_path) in table_artifacts.items()
            if csv_path is not None and meta_path is not None
        )
    )
    return WorkflowInputContract(
        data_root=root,
        census_pumf_root=census_root,
        housing_survey_root=housing_root,
        individual_pumf_candidates=find_individual_pumf_candidates(root, census_pumf_root=census_root),
        household_pumf_candidates=find_household_pumf_candidates(root, census_pumf_root=census_root),
        housing_survey_candidates=find_housing_survey_candidates(housing_survey_root=housing_root),
        available_census_tables=available_tables,
    )


def summarize_workflow_input_contract(
    data_root: str | Path,
    *,
    census_pumf_root: str | Path | None = None,
    housing_survey_root: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize required inputs as a one-row-per-artifact audit table."""
    contract = build_workflow_input_contract(
        data_root,
        census_pumf_root=census_pumf_root,
        housing_survey_root=housing_survey_root,
    )
    required_tables = [
        ("age_sex_core", "age_sex_core"),
        ("education_detailed", "education_detailed"),
        ("labour_detailed", "labour_detailed"),
        ("income_detailed", "income_detailed"),
        ("household_type_size_detailed", "household_type_size_detailed"),
        ("dwelling_characteristics", "dwelling_characteristics"),
        ("housing", "housing"),
        ("immigration_citizenship", "immigration_citizenship"),
        ("commute", "commute"),
    ]

    rows = [
        {
            "artifact_type": "individual_pumf",
            "artifact_name": "person_seed_microdata",
            "required": True,
            "found": bool(contract.individual_pumf_candidates),
            "details": str(contract.individual_pumf_candidates[0]) if contract.individual_pumf_candidates else None,
        },
        {
            "artifact_type": "household_pumf",
            "artifact_name": "household_seed_microdata",
            "required": True,
            "found": bool(contract.household_pumf_candidates),
            "details": str(contract.household_pumf_candidates[0]) if contract.household_pumf_candidates else None,
        },
        {
            "artifact_type": "housing_survey",
            "artifact_name": "chs_household_housing_seed",
            "required": False,
            "found": bool(contract.housing_survey_candidates),
            "details": str(contract.housing_survey_candidates[0]) if contract.housing_survey_candidates else None,
        },
    ]

    available = set(contract.available_census_tables)
    table_artifacts = resolve_da_table_artifacts(data_root)
    for artifact_name, table_key in required_tables:
        csv_path, _meta_path = table_artifacts[table_key]
        rows.append(
            {
                "artifact_type": "da_census_table",
                "artifact_name": artifact_name,
                "required": True,
                "found": table_key in available,
                "details": str(csv_path) if table_key in available and csv_path is not None else None,
            }
        )
    return pd.DataFrame(rows)


def build_workflow_attribute_blueprint(*, use_housing_survey: bool = False) -> pd.DataFrame:
    """Describe the supported workflow attributes and where each one comes from."""
    household_seed_source = "hierarchical_pumf_plus_chs" if use_housing_survey else "household_pumf"
    household_status = "implemented_harmonized_with_chs" if use_housing_survey else "implemented_harmonized"

    rows = [
        {
            "attribute": "age_group",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "age_sex_core",
            "seed_source": "individual_pumf",
            "status": "implemented",
        },
        {
            "attribute": "sex",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "age_sex_core",
            "seed_source": "individual_pumf",
            "status": "implemented",
        },
        {
            "attribute": "education_level",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "education_detailed",
            "seed_source": "individual_pumf",
            "status": "implemented_coarse",
        },
        {
            "attribute": "labour_force_status",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "labour_detailed",
            "seed_source": "individual_pumf",
            "status": "implemented_coarse",
        },
        {
            "attribute": "household_income",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "income_detailed",
            "seed_source": "individual_pumf",
            "status": "implemented_coarse",
        },
        {
            "attribute": "citizenship_status",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "immigration_citizenship",
            "seed_source": "individual_pumf",
            "status": "implemented_harmonized",
        },
        {
            "attribute": "immigrant_status",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "immigration_citizenship",
            "seed_source": "individual_pumf",
            "status": "implemented_harmonized",
        },
        {
            "attribute": "commute_mode",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "commute",
            "seed_source": "individual_pumf",
            "status": "implemented_harmonized",
        },
        {
            "attribute": "commute_duration",
            "target_level": "person",
            "assignment_method": "person_ipf",
            "census_source": "commute",
            "seed_source": "individual_pumf",
            "status": "implemented_harmonized",
        },
        {
            "attribute": "household_size",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "household_type_size_detailed",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "household_type",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "household_type_size_detailed",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "dwelling_type",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "dwelling_characteristics",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "tenure",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "bedrooms",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "housing_suitability",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": "supported_but_not_in_default_runner",
        },
        {
            "attribute": "period_built",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "dwelling_condition",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": household_status,
        },
        {
            "attribute": "core_housing_need",
            "target_level": "household",
            "assignment_method": "household_ipf",
            "census_source": "housing",
            "seed_source": household_seed_source,
            "status": household_status,
        },
    ]
    return pd.DataFrame(rows)


def build_workflow_step_blueprint(*, use_housing_survey: bool = False) -> pd.DataFrame:
    """Describe the high-level workflow steps in the supported synthesis pipeline."""
    household_seed_phrase = (
        "the 2021 Census Hierarchical PUMF plus the 2022 CHS PUMF"
        if use_housing_survey
        else "the 2021 Census Hierarchical PUMF"
    )
    return pd.DataFrame(
        [
            {
                "step_order": 1,
                "step_name": "load_inputs",
                "unit": "shared",
                "description": "Load DA census marginals plus individual and household seed microdata from the declared input roots.",
            },
            {
                "step_order": 2,
                "step_name": "prepare_person_seed",
                "unit": "person",
                "description": "Map the 2021 Census Individuals PUMF into the harmonized person categories used by the DA person marginals.",
            },
            {
                "step_order": 3,
                "step_name": "prepare_household_seed",
                "unit": "household",
                "description": f"Map {household_seed_phrase} into harmonized housing and household categories used by the DA household marginals.",
            },
            {
                "step_order": 4,
                "step_name": "synthesize_people",
                "unit": "person",
                "description": "Run the person-level IPF/QISI stage for age, sex, education, labour, income, citizenship, immigration, and commute variables.",
            },
            {
                "step_order": 5,
                "step_name": "synthesize_households",
                "unit": "household",
                "description": "Run the household-level IPF/QISI stage for household type, size, dwelling type, tenure, bedrooms, suitability, period built, condition, and core need.",
            },
            {
                "step_order": 6,
                "step_name": "assemble_people_into_households",
                "unit": "joint",
                "description": "Match synthetic people to synthetic households using compatibility rules instead of heuristic post-assignment.",
            },
            {
                "step_order": 7,
                "step_name": "validate_outputs",
                "unit": "joint",
                "description": "Evaluate both marginal fit and pairwise structure preservation for the harmonized workflow attributes.",
            },
        ]
    )
