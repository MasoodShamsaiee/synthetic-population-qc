"""Unified joint-fit household-donor synthesis for the supported workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_population_qc.context_tables import load_context_tables
from synthetic_population_qc.enrichment import HOUSEHOLD_ATTR_LABELS, PERSON_ATTR_LABELS
from synthetic_population_qc.ingest.preprocess import load_processed_context_tables
from synthetic_population_qc.public_schema import AGE_GROUP_BOUNDS, public_attr_name, public_value_label, public_value_series
from synthetic_population_qc.reporting import build_results_summary
from synthetic_population_qc.runs.bundle import load_processed_artifacts
from synthetic_population_qc.scope_selection import resolve_da_scope_codes
from synthetic_population_qc.seed_preparation import (
    _derive_household_type_from_members,
    _map_citizenship_status,
    _map_commute_duration,
    _map_commute_mode_group,
    _map_core_need_census,
    _map_dwelling_type_census,
    _map_immigrant_status,
    _map_repair_census,
    _map_tenure_census,
    _map_bedrooms_census,
    _map_built_census,
    _prepare_core_seed,
    load_census_hierarchical_pumf,
    prepare_hierarchical_household_seed,
)
from synthetic_population_qc.workflow_inputs import find_household_pumf_candidates


HOUSEHOLD_TYPE_TARGET_COLUMNS = {
    "couple_with_children": (
        "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / "
        "Couple-family households / With children"
    ),
    "couple_without_children": (
        "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / "
        "Couple-family households / Without children"
    ),
    "one_parent": (
        "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / "
        "One-parent-family households"
    ),
    "one_person": "Households by type / Total - Household type - 100% data / One-person households",
    "other": "Households by type / Total - Household type - 100% data / Two-or-more-person non-census-family households",
}

EDUCATION_TARGET_COLUMNS = {
    1684: (
        "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private "
        "households - 25% sample data ; Both sexes / No certificate, diploma or degree ; Both sexes"
    ),
    1685: (
        "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private "
        "households - 25% sample data ; Both sexes / High (secondary) school diploma or equivalency certificate ; Both sexes"
    ),
    1686: (
        "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private "
        "households - 25% sample data ; Both sexes / Postsecondary certificate, diploma or degree ; Both sexes"
    ),
}

LABOUR_TARGET_COLUMNS = {
    1867: (
        "Labour - Total Sex / Total - Population aged 15 years and over by labour force status - 25% sample data ; Both sexes / In "
        "the labour force ; Both sexes / Employed ; Both sexes"
    ),
    1868: (
        "Labour - Total Sex / Total - Population aged 15 years and over by labour force status - 25% sample data ; Both sexes / In "
        "the labour force ; Both sexes / Unemployed ; Both sexes"
    ),
}

INCOME_TARGET_COLUMNS = {
    695: [
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / Under $10,000 (including loss) ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $10,000 to $19,999 ; Both sexes"
        ),
    ],
    697: [
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $20,000 to $29,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $30,000 to $39,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $40,000 to $49,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $50,000 to $59,999 ; Both sexes"
        ),
    ],
    701: [
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $60,000 to $69,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $70,000 to $79,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $80,000 to $89,999 ; Both sexes"
        ),
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $90,000 to $99,999 ; Both sexes"
        ),
    ],
    705: [
        (
            "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private "
            "households - 100% data ; Both sexes / With after-tax income ; Both sexes / $100,000 and over ; Both sexes"
        ),
    ],
}

AGE_SEX_TARGET_COLUMNS = {
    "male": {
        10: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 0 to 4 years ; Males"],
        11: [
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 5 to 9 years ; Males",
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 10 to 14 years ; Males",
        ],
        12: [
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 15 to 19 years ; Males",
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 20 to 24 years ; Males",
        ],
        14: [
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 25 to 29 years ; Males",
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 30 to 34 years ; Males",
        ],
        15: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 35 to 39 years ; Males"],
        16: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 40 to 44 years ; Males"],
        17: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 45 to 49 years ; Males"],
        18: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 50 to 54 years ; Males"],
        19: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 55 to 59 years ; Males"],
        20: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 60 to 64 years ; Males"],
        21: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 65 to 69 years ; Males"],
        22: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 70 to 74 years ; Males"],
        23: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 75 to 79 years ; Males"],
        25: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 80 to 84 years ; Males"],
        26: ["Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 85 years and over ; Males"],
    },
    "female": {
        10: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 0 to 4 years ; Females"],
        11: [
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 5 to 9 years ; Females",
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 10 to 14 years ; Females",
        ],
        12: [
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 15 to 19 years ; Females",
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 20 to 24 years ; Females",
        ],
        14: [
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 25 to 29 years ; Females",
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 30 to 34 years ; Females",
        ],
        15: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 35 to 39 years ; Females"],
        16: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 40 to 44 years ; Females"],
        17: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 45 to 49 years ; Females"],
        18: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 50 to 54 years ; Females"],
        19: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 55 to 59 years ; Females"],
        20: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 60 to 64 years ; Females"],
        21: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 65 to 69 years ; Females"],
        22: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 70 to 74 years ; Females"],
        23: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 75 to 79 years ; Females"],
        25: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 80 to 84 years ; Females"],
        26: ["Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 85 years and over ; Females"],
    },
}

HOUSEHOLD_SIZE_TARGET_COLUMNS = {
    "1": "Households by type / Total - Household type - 100% data / One-person households",
    "non_family_2plus": "Households by type / Total - Household type - 100% data / Two-or-more-person non-census-family households",
    "2": "Family characteristics / Total - Census families in private households by family size - 100% data / 2 persons",
    "3": "Family characteristics / Total - Census families in private households by family size - 100% data / 3 persons",
    "4": "Family characteristics / Total - Census families in private households by family size - 100% data / 4 persons",
    "5plus": "Family characteristics / Total - Census families in private households by family size - 100% data / 5 or more persons",
}

TIER_WEIGHTS = {"tier_1": 4.0, "tier_2": 2.0, "tier_3": 1.0}
TIER_THRESHOLDS = {"tier_1": 0.02, "tier_2": 0.05, "tier_3": 0.10}
JOINT_REFERENCE_SOURCE = "smoothed_direct_da"
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_IMPROVEMENT_TOL = 1e-4
DERIVED_OUTPUT_ATTRIBUTES = ("family_status",)


@dataclass(frozen=True)
class JointControlSpec:
    """Static description of one unified control used by the joint fit."""

    attribute: str
    unit: str
    tier: str
    donor_column: str
    categories: tuple[object, ...]
    target_table: str | None = None
    restricted_universe: str = "all"

    @property
    def tier_weight(self) -> float:
        return float(TIER_WEIGHTS[self.tier])


@dataclass(frozen=True)
class JointWorkflowArtifacts:
    """Typed file outputs produced by one joint-fit workflow execution."""

    output_dir: Path
    people_parquet: Path
    households_parquet: Path
    workflow_plan_json: Path
    assignment_route_csv: Path
    household_coherence_csv: Path
    summary_csv: Path
    details_csv: Path
    results_summary_csv: Path
    support_classification_csv: Path
    sparse_handling_csv: Path
    metadata_json: Path


JOINT_CONTROL_SPECS: tuple[JointControlSpec, ...] = (
    JointControlSpec("sex", "person", "tier_1", "sex", ("female", "male"), "age_sex_core"),
    JointControlSpec("age_group", "person", "tier_1", "age_group", tuple(sorted(AGE_GROUP_BOUNDS)), "age_sex_core"),
    JointControlSpec(
        "household_type",
        "household",
        "tier_1",
        "household_type",
        tuple(HOUSEHOLD_TYPE_TARGET_COLUMNS),
        "household_type_size_detailed",
    ),
    JointControlSpec("household_size", "household", "tier_1", "household_size", ("1", "2", "3", "4", "5plus"), "household_type_size_detailed"),
    JointControlSpec("education_level", "person", "tier_2", "education_level", (1684, 1685, 1686), "education_detailed", "age_25_64"),
    JointControlSpec("labour_force_status", "person", "tier_2", "labour_force_status", (1867, 1868, 1869), "labour_detailed", "age_15_plus"),
    JointControlSpec("household_income", "person", "tier_2", "household_income", (695, 697, 701, 705), "income_detailed", "age_15_plus"),
    JointControlSpec("tenure", "household", "tier_2", "tenure", ("owner", "renter_or_band"), "housing"),
    JointControlSpec(
        "dwelling_type",
        "household",
        "tier_2",
        "dwelling_type",
        ("single_detached_house", "apartment", "other_dwelling"),
        "dwelling_characteristics",
    ),
    JointControlSpec(
        "citizenship_status",
        "person",
        "tier_3",
        "citizenship_status",
        ("canadian_citizen", "not_canadian_citizen"),
        "immigration_citizenship",
    ),
    JointControlSpec(
        "immigrant_status",
        "person",
        "tier_3",
        "immigrant_status",
        ("non_immigrant", "immigrant", "non_permanent_resident"),
        "immigration_citizenship",
    ),
    JointControlSpec(
        "commute_mode",
        "person",
        "tier_3",
        "commute_mode",
        ("private_vehicle", "public_transit", "active_transport", "other_method"),
        "commute",
        "employed",
    ),
    JointControlSpec(
        "commute_duration",
        "person",
        "tier_3",
        "commute_duration",
        ("lt_15_min", "15_to_29_min", "30_to_44_min", "45_to_59_min", "60_plus_min"),
        "commute",
        "employed",
    ),
    JointControlSpec(
        "bedrooms",
        "household",
        "tier_3",
        "bedrooms",
        ("no_bedroom", "1_bedroom", "2_bedrooms", "3_bedrooms", "4plus_bedrooms"),
        "housing",
    ),
    JointControlSpec(
        "period_built",
        "household",
        "tier_3",
        "period_built",
        (
            "1945_or_before",
            "1946_to_1960",
            "1961_to_1970",
            "1971_to_1980",
            "1981_to_1990",
            "1991_to_1995",
            "1996_to_2000",
            "2001_to_2005",
            "2006_to_2010",
            "2011_to_2015",
            "2016_to_2021",
        ),
        "housing",
    ),
    JointControlSpec(
        "dwelling_condition",
        "household",
        "tier_3",
        "dwelling_condition",
        ("regular_maintenance", "minor_repairs", "major_repairs"),
        "housing",
    ),
    JointControlSpec(
        "core_housing_need",
        "household",
        "tier_3",
        "core_housing_need",
        ("not_in_core_need", "in_core_need"),
        "housing",
    ),
)


def _resolve_existing_raw_root(root: str | Path, *, leaf_name: str | None = None) -> Path:
    """Resolve a declared raw-data root onto an existing directory when possible."""
    declared = Path(root)
    candidates = [declared]
    name = leaf_name or declared.name
    candidates.extend(
        [
            declared / "data" / "raw" / name,
            declared.parent / "data" / "raw" / name,
            declared / name,
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return declared


def _largest_remainder_to_total(targets: dict[object, float], total_n: int) -> dict[object, int]:
    """Integerize fractional targets while preserving the requested total."""
    if total_n <= 0:
        return {key: 0 for key in targets}
    vals = {key: max(0.0, float(value)) for key, value in targets.items()}
    current_total = sum(vals.values())
    if current_total > 0:
        vals = {key: (value / current_total) * float(total_n) for key, value in vals.items()}
    else:
        vals = {key: float(total_n) / float(len(vals)) for key in vals} if vals else {}
    floors = {key: int(np.floor(value)) for key, value in vals.items()}
    remainder = total_n - sum(floors.values())
    order = sorted(vals, key=lambda key: vals[key] - floors[key], reverse=True)
    for idx in range(max(remainder, 0)):
        floors[order[idx % len(order)]] += 1
    return floors


def _get_da_row(df: pd.DataFrame, da_code: str) -> pd.Series | None:
    """Return the first row matching one DA code from a context table."""
    if df.empty or "da_code" not in df.columns:
        return None
    match = df.loc[df["da_code"].astype(str).str.strip() == str(da_code).strip()]
    if match.empty:
        return None
    return match.iloc[0]


def _sum_columns(row: pd.Series | None, columns: list[str]) -> float:
    """Sum a set of numeric columns from one context-table row."""
    if row is None:
        return 0.0
    total = 0.0
    for column in columns:
        total += float(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").fillna(0.0).iloc[0])
    return total


def _household_total_from_context(context_tables: dict[str, pd.DataFrame], da_code: str) -> int:
    """Return the total occupied private dwellings for one DA when available."""
    dwelling_row = _get_da_row(context_tables.get("dwelling_characteristics", pd.DataFrame()), da_code)
    if dwelling_row is not None:
        total = pd.to_numeric(
            pd.Series(
                [dwelling_row.get("Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data")]
            ),
            errors="coerce",
        ).fillna(0.0).iloc[0]
        if total > 0:
            return int(round(float(total)))
    housing_row = _get_da_row(context_tables.get("housing", pd.DataFrame()), da_code)
    if housing_row is not None:
        total = _sum_columns(housing_row, list(HOUSEHOLD_ATTR_LABELS["tenure"].values()))
        if total > 0:
            return int(round(total))
    return 0


def _pooled_row(table: pd.DataFrame, da_codes: list[str]) -> pd.Series | None:
    """Aggregate one context table over the selected DAs for smoothing."""
    if table.empty or "da_code" not in table.columns:
        return None
    subset = table.loc[table["da_code"].astype(str).isin(da_codes)].copy()
    if subset.empty:
        return None
    numeric_cols = [col for col in subset.columns if col not in {"GEO UID", "DA name", "da_code"}]
    values = {col: float(pd.to_numeric(subset[col], errors="coerce").fillna(0.0).sum()) for col in numeric_cols}
    return pd.Series(values)


def _broader_row(table: pd.DataFrame) -> pd.Series | None:
    """Aggregate one context table across all available rows for broad smoothing."""
    if table.empty:
        return None
    numeric_cols = [col for col in table.columns if col not in {"GEO UID", "DA name", "da_code"}]
    values = {col: float(pd.to_numeric(table[col], errors="coerce").fillna(0.0).sum()) for col in numeric_cols}
    return pd.Series(values)


def _reverse_grouped_commute(value: object) -> str | None:
    """Collapse raw commute group labels onto the supported public output labels."""
    value_str = str(value).strip()
    if value_str == "car_truck_van":
        return "private_vehicle"
    if value_str == "public_transit":
        return "public_transit"
    if value_str in {"walked", "bicycle"}:
        return "active_transport"
    if value_str == "other_method":
        return "other_method"
    return None


def _reverse_commute_duration(value: object) -> str | None:
    """Collapse raw commute-duration labels onto the supported public output labels."""
    value_str = str(value).strip()
    if value_str == "60plus_min":
        return "60_plus_min"
    if value_str in {"lt_15_min", "15_to_29_min", "30_to_44_min", "45_to_59_min"}:
        return value_str
    return None


def _scalar_grouped_commute(value: object) -> str | None:
    """Map one raw commute-mode code to the grouped public commute label."""
    grouped = _map_commute_mode_group(pd.Series([value])).iloc[0]
    return _reverse_grouped_commute(grouped)


def _scalar_commute_duration(value: object) -> str | None:
    """Map one raw commute-duration code to the public commute-duration label."""
    grouped = _map_commute_duration(pd.Series([value])).iloc[0]
    return _reverse_commute_duration(grouped)


def _prepare_joint_household_donors(
    census_pumf_root: str | Path,
    *,
    province: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build household-level donor rows plus attached member rows."""
    resolved_root = _resolve_existing_raw_root(census_pumf_root, leaf_name="PUMF")
    candidates = find_household_pumf_candidates(resolved_root, census_pumf_root=resolved_root)
    csv_candidates = [path for path in candidates if Path(path).suffix.lower() == ".csv"]
    if not csv_candidates:
        raise FileNotFoundError(
            f"No hierarchical census PUMF CSV candidate found under {resolved_root}"
        )
    raw = load_census_hierarchical_pumf(csv_candidates[0], province=province)
    raw = raw.copy()
    raw["HHSIZE"] = raw.groupby("HH_ID", sort=False)["HH_ID"].transform("size")
    core = _prepare_core_seed(raw, gender_col="GENDER", totinc_col="TOTINC")
    household_types = _derive_household_type_from_members(raw)
    members = core.merge(household_types, on="HH_ID", how="left")
    citizen_col = "CITIZEN" if "CITIZEN" in members.columns else "Citizen"
    immigrant_col = "IMMSTAT" if "IMMSTAT" in members.columns else "ImmStat"
    commute_mode_col = "MODE" if "MODE" in members.columns else "Mode"
    commute_duration_col = "PWDUR" if "PWDUR" in members.columns else "PwDur"
    members["household_size"] = (
        pd.to_numeric(members["n_members"], errors="coerce").clip(lower=1).fillna(1).astype(int).clip(upper=5).map(
            {1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"}
        )
    )
    member_seed = pd.DataFrame(
        {
            "household_id": members["HH_ID"].astype(str),
            "person_id": members["PP_ID"].astype(str),
            "weight": pd.to_numeric(members["WEIGHT"], errors="coerce").fillna(0.0),
            "sex": members["Gender"].map({1: "male", 2: "female"}),
            "age_group": pd.to_numeric(members["agegrp"], errors="coerce"),
            "education_level": pd.to_numeric(members["hdgree"], errors="coerce"),
            "labour_force_status": pd.to_numeric(members["lfact"], errors="coerce"),
            "household_income": pd.to_numeric(members["TotInc"], errors="coerce"),
            "family_status": pd.to_numeric(members["cfstat"], errors="coerce"),
            "citizenship_status": _map_citizenship_status(members[citizen_col]),
            "immigrant_status": _map_immigrant_status(members[immigrant_col]),
            "commute_mode": members[commute_mode_col].map(_scalar_grouped_commute),
            "commute_duration": members[commute_duration_col].map(_scalar_commute_duration),
            "household_type": members["household_type"],
            "household_size": members["household_size"],
        }
    ).reset_index(drop=True)

    household_seed = prepare_hierarchical_household_seed(raw)[
        [
            "household_id",
            "reference_person_id",
            "weight",
            "household_type",
            "household_size",
            "dwelling_type",
            "tenure",
            "bedrooms",
            "period_built",
            "dwelling_condition",
            "core_housing_need",
        ]
    ].copy()
    return household_seed.reset_index(drop=True), member_seed


def _eligible_members(member_df: pd.DataFrame, restricted_universe: str) -> pd.DataFrame:
    """Filter donor members onto the requested universe for one control."""
    if restricted_universe == "all":
        return member_df
    if restricted_universe == "age_15_plus":
        return member_df.loc[pd.to_numeric(member_df["age_group"], errors="coerce").isin([12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26])]
    if restricted_universe == "age_25_64":
        return member_df.loc[pd.to_numeric(member_df["age_group"], errors="coerce").isin([14, 15, 16, 17, 18, 19, 20])]
    if restricted_universe == "employed":
        return member_df.loc[pd.to_numeric(member_df["labour_force_status"], errors="coerce").eq(1867)]
    raise ValueError(f"Unsupported restricted universe: {restricted_universe}")


def _donor_prior_counts(
    spec: JointControlSpec,
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
    target_total: int,
) -> dict[object, int]:
    """Build one prior target from donor distributions when no direct DA total exists."""
    if spec.unit == "household":
        counts = household_donors[spec.donor_column].value_counts(dropna=True).to_dict()
    else:
        eligible = _eligible_members(member_donors, spec.restricted_universe)
        counts = eligible[spec.donor_column].value_counts(dropna=True).to_dict()
    return _largest_remainder_to_total({category: float(counts.get(category, 0.0)) for category in spec.categories}, target_total)


def _donor_share(series: pd.Series, categories: tuple[object, ...]) -> dict[object, float]:
    """Return normalized donor shares over the requested categories."""
    counts = series.value_counts(dropna=True).to_dict()
    total = float(sum(float(counts.get(category, 0.0)) for category in categories))
    if total <= 0:
        uniform = 1.0 / float(len(categories)) if categories else 0.0
        return {category: uniform for category in categories}
    return {category: float(counts.get(category, 0.0)) / total for category in categories}


def _split_total_by_share(total: int, share: dict[object, float]) -> dict[object, int]:
    """Allocate one integer total according to a share map."""
    if total <= 0:
        return {category: 0 for category in share}
    return _largest_remainder_to_total({category: float(weight) for category, weight in share.items()}, int(total))


def _other_household_size_share(household_donors: pd.DataFrame | None) -> dict[str, float]:
    """Estimate size shares among non-census-family donor households."""
    categories = ("2", "3", "4", "5plus")
    if household_donors is None or household_donors.empty:
        return {category: 0.25 for category in categories}
    subset = household_donors.loc[household_donors["household_type"].astype(str).eq("other"), "household_size"]
    return _donor_share(subset.astype(object), categories)


def _extract_direct_counts(
    spec: JointControlSpec,
    row: pd.Series | None,
    income_total_15_plus: int | None = None,
    *,
    household_donors: pd.DataFrame | None = None,
    member_donors: pd.DataFrame | None = None,
) -> dict[object, int]:
    """Extract one DA target distribution directly from labeled context tables."""
    if row is None:
        return {category: 0 for category in spec.categories}
    if spec.attribute == "sex":
        return {
            sex: int(round(sum(_sum_columns(row, columns) for columns in AGE_SEX_TARGET_COLUMNS[sex].values())))
            for sex in spec.categories
        }
    if spec.attribute == "age_group":
        return {
            category: int(
                round(
                    _sum_columns(row, AGE_SEX_TARGET_COLUMNS["male"].get(int(category), []))
                    + _sum_columns(row, AGE_SEX_TARGET_COLUMNS["female"].get(int(category), []))
                )
            )
            for category in spec.categories
        }
    if spec.attribute == "household_type":
        return {category: int(round(float(pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").fillna(0.0).iloc[0]))) for category, column in HOUSEHOLD_TYPE_TARGET_COLUMNS.items()}
    if spec.attribute == "household_size":
        non_family_two_plus = int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["non_family_2plus"]])))
        non_family_split = _split_total_by_share(non_family_two_plus, _other_household_size_share(household_donors))
        counts = {
            "1": int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["1"]]))),
            "2": int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["2"]]))),
            "3": int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["3"]]))),
            "4": int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["4"]]))),
            "5plus": int(round(_sum_columns(row, [HOUSEHOLD_SIZE_TARGET_COLUMNS["5plus"]]))),
        }
        for category in ("2", "3", "4", "5plus"):
            counts[category] += int(non_family_split.get(category, 0))
        return counts
    if spec.attribute == "dwelling_type":
        apt_total = _sum_columns(
            row,
            [
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["duplex_apartment"],
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["lowrise_apartment"],
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["highrise_apartment"],
            ],
        )
        other_total = _sum_columns(
            row,
            [
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["semi_detached"],
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["row_house"],
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["other_single_attached"],
                HOUSEHOLD_ATTR_LABELS["dwelling_type"]["movable_dwelling"],
            ],
        )
        return {
            "single_detached_house": int(round(_sum_columns(row, [HOUSEHOLD_ATTR_LABELS["dwelling_type"]["single_detached"]]))),
            "apartment": int(round(apt_total)),
            "other_dwelling": int(round(other_total)),
        }
    if spec.attribute in {"tenure", "bedrooms", "core_housing_need"}:
        label_map = HOUSEHOLD_ATTR_LABELS[spec.attribute]
        return {category: int(round(_sum_columns(row, [label_map.get("renter") if spec.attribute == "tenure" and category == "renter_or_band" else label_map.get(category)]))) for category in spec.categories}
    if spec.attribute == "dwelling_condition":
        label_map = HOUSEHOLD_ATTR_LABELS["dwelling_condition"]
        regular_minor = int(round(_sum_columns(row, [label_map["regular_or_minor_repairs"]])))
        major = int(round(_sum_columns(row, [label_map["major_repairs_needed"]])))
        # The DA housing table combines regular maintenance and minor repairs;
        # use an even proxy split so the joint fit can keep this attribute on
        # the direct-DA target surface until a finer public table is added.
        regular_count = int(round(regular_minor / 2.0))
        regular_count = max(min(regular_count, regular_minor), 0)
        minor_count = max(regular_minor - regular_count, 0)
        return {
            "regular_maintenance": regular_count,
            "minor_repairs": minor_count,
            "major_repairs": major,
        }
    if spec.attribute == "period_built":
        source_map = {
            "1945_or_before": ["1960_or_before"],
            "1946_to_1960": ["1960_or_before"],
            "1961_to_1970": ["1961_to_1980"],
            "1971_to_1980": ["1961_to_1980"],
            "1981_to_1990": ["1981_to_1990"],
            "1991_to_1995": ["1991_to_2000"],
            "1996_to_2000": ["1991_to_2000"],
            "2001_to_2005": ["2001_to_2005"],
            "2006_to_2010": ["2006_to_2010"],
            "2011_to_2015": ["2011_to_2015"],
            "2016_to_2021": ["2016_to_2021"],
        }
        label_map = HOUSEHOLD_ATTR_LABELS["period_built"]
        counts: dict[object, int] = {}
        for category, raw_keys in source_map.items():
            counts[category] = int(round(_sum_columns(row, [label_map[key] for key in raw_keys if key in label_map])))
        return counts
    if spec.attribute == "citizenship_status":
        return {category: int(round(_sum_columns(row, [PERSON_ATTR_LABELS["citizenship_status"][category]]))) for category in spec.categories}
    if spec.attribute == "immigrant_status":
        return {category: int(round(_sum_columns(row, [PERSON_ATTR_LABELS["immigrant_status"][category]]))) for category in spec.categories}
    if spec.attribute == "commute_mode":
        counts = {
            "private_vehicle": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_mode"]["car_truck_van"]]))),
            "public_transit": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_mode"]["public_transit"]]))),
            "active_transport": int(
                round(
                    _sum_columns(
                        row,
                        [PERSON_ATTR_LABELS["commute_mode"]["walked"], PERSON_ATTR_LABELS["commute_mode"]["bicycle"]],
                    )
                )
            ),
            "other_method": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_mode"]["other_method"]]))),
        }
        return counts
    if spec.attribute == "commute_duration":
        return {
            "lt_15_min": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_duration"]["lt_15_min"]]))),
            "15_to_29_min": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_duration"]["15_to_29_min"]]))),
            "30_to_44_min": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_duration"]["30_to_44_min"]]))),
            "45_to_59_min": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_duration"]["45_to_59_min"]]))),
            "60_plus_min": int(round(_sum_columns(row, [PERSON_ATTR_LABELS["commute_duration"]["60plus_min"]]))),
        }
    if spec.attribute == "education_level":
        return {category: int(round(_sum_columns(row, [column]))) for category, column in EDUCATION_TARGET_COLUMNS.items()}
    if spec.attribute == "household_income":
        return {category: int(round(_sum_columns(row, columns))) for category, columns in INCOME_TARGET_COLUMNS.items()}
    if spec.attribute == "labour_force_status":
        employed = int(round(_sum_columns(row, [LABOUR_TARGET_COLUMNS[1867]])))
        unemployed = int(round(_sum_columns(row, [LABOUR_TARGET_COLUMNS[1868]])))
        total_15_plus = max(int(income_total_15_plus or 0), employed + unemployed)
        return {
            1867: employed,
            1868: unemployed,
            1869: max(total_15_plus - employed - unemployed, 0),
        }
    return {category: 0 for category in spec.categories}


def _estimate_target_total(
    spec: JointControlSpec,
    *,
    household_total: int,
    direct_counts: dict[object, int],
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
    person_total_hint: int,
) -> int:
    """Estimate the total number of eligible items for one control when needed."""
    direct_total = int(sum(int(value) for value in direct_counts.values()))
    if direct_total > 0:
        return direct_total
    if spec.unit == "household":
        return int(household_total)
    if spec.restricted_universe == "all":
        return int(person_total_hint)
    donor_household_weights = pd.to_numeric(household_donors["weight"], errors="coerce").fillna(0.0)
    if donor_household_weights.sum() <= 0:
        donor_household_weights = pd.Series(np.ones(len(household_donors)), index=household_donors.index)
    if spec.restricted_universe == "age_15_plus":
        eligible = _eligible_members(member_donors, "age_15_plus")
    elif spec.restricted_universe == "age_25_64":
        eligible = _eligible_members(member_donors, "age_25_64")
    elif spec.restricted_universe == "employed":
        eligible = _eligible_members(member_donors, "employed")
    else:
        eligible = member_donors
    if eligible.empty:
        return 0
    per_household = eligible.groupby("household_id").size().reindex(household_donors["household_id"]).fillna(0.0).to_numpy(dtype=float)
    mean_count = float(np.average(per_household, weights=donor_household_weights.to_numpy(dtype=float)))
    return int(round(mean_count * float(household_total)))


def _smoothed_target_for_da(
    spec: JointControlSpec,
    *,
    da_code: str,
    da_codes: list[str],
    context_tables: dict[str, pd.DataFrame],
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
    household_total: int,
    person_total_hint: int,
) -> tuple[dict[object, int], dict[str, object]]:
    """Build one DA-level target with provenance-aware smoothing."""
    table = context_tables.get(spec.target_table, pd.DataFrame()) if spec.target_table is not None else pd.DataFrame()
    direct_row = _get_da_row(table, da_code) if spec.target_table is not None else None
    pooled = _pooled_row(table, da_codes) if spec.target_table is not None else None
    broader = _broader_row(table) if spec.target_table is not None else None
    income_total_15_plus = None
    if spec.attribute == "labour_force_status":
        income_row = _get_da_row(context_tables.get("income_detailed", pd.DataFrame()), da_code)
        if income_row is not None:
            income_total_15_plus = int(
                round(
                    sum(
                        _extract_direct_counts(
                            JOINT_CONTROL_SPECS[6],
                            income_row,
                            household_donors=household_donors,
                            member_donors=member_donors,
                        ).values()
                    )
                )
            )

    direct_counts = _extract_direct_counts(
        spec,
        direct_row,
        income_total_15_plus=income_total_15_plus,
        household_donors=household_donors,
        member_donors=member_donors,
    )
    pooled_counts = _extract_direct_counts(
        spec,
        pooled,
        household_donors=household_donors,
        member_donors=member_donors,
    )
    broader_counts = _extract_direct_counts(
        spec,
        broader,
        household_donors=household_donors,
        member_donors=member_donors,
    )
    target_total = _estimate_target_total(
        spec,
        household_total=household_total,
        direct_counts=direct_counts,
        household_donors=household_donors,
        member_donors=member_donors,
        person_total_hint=person_total_hint,
    )

    direct_total = int(sum(direct_counts.values()))
    pooled_total = int(sum(pooled_counts.values()))
    broader_total = int(sum(broader_counts.values()))
    if direct_total > 0:
        effective = _largest_remainder_to_total({category: float(direct_counts.get(category, 0)) for category in spec.categories}, target_total)
        provenance = "direct_da"
    elif pooled_total > 0:
        effective = _largest_remainder_to_total({category: float(pooled_counts.get(category, 0)) for category in spec.categories}, target_total)
        provenance = "selected_da_pool"
    elif broader_total > 0:
        effective = _largest_remainder_to_total({category: float(broader_counts.get(category, 0)) for category in spec.categories}, target_total)
        provenance = "broader_geography"
    else:
        effective = _donor_prior_counts(spec, household_donors, member_donors, target_total)
        provenance = "seed_prior"
    smoothing_mode = "direct" if provenance == "direct_da" else f"borrow_{provenance}"
    return effective, {
        "attribute": spec.attribute,
        "unit": spec.unit,
        "control_tier": spec.tier,
        "target_source": spec.target_table or "seed_prior",
        "target_provenance": provenance,
        "smoothing_mode": smoothing_mode,
        "raw_target_total": direct_total,
        "effective_target_total": int(sum(effective.values())),
    }


def _household_incidence_matrix(
    spec: JointControlSpec,
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
) -> np.ndarray:
    """Build a donor-by-category incidence matrix for one unified control."""
    donor_ids = household_donors["household_id"].astype(str).tolist()
    if spec.unit == "household":
        matrix = np.zeros((len(donor_ids), len(spec.categories)), dtype=float)
        values = household_donors[spec.donor_column].astype(object).tolist()
        for row_idx, value in enumerate(values):
            for col_idx, category in enumerate(spec.categories):
                matrix[row_idx, col_idx] = 1.0 if value == category else 0.0
        return matrix

    eligible = _eligible_members(member_donors, spec.restricted_universe)
    grouped = (
        eligible.groupby(["household_id", spec.donor_column], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    matrix = np.zeros((len(donor_ids), len(spec.categories)), dtype=float)
    donor_pos = {donor_id: idx for idx, donor_id in enumerate(donor_ids)}
    category_pos = {category: idx for idx, category in enumerate(spec.categories)}
    for row in grouped.itertuples(index=False):
        donor_id = str(getattr(row, "household_id"))
        category = getattr(row, spec.donor_column)
        if donor_id in donor_pos and category in category_pos:
            matrix[donor_pos[donor_id], category_pos[category]] = float(getattr(row, "n"))
    return matrix


def _weighted_fit_error(current: np.ndarray, target: np.ndarray) -> float:
    """Return one normalized absolute error for a target vector."""
    total = float(target.sum())
    if total <= 0:
        return 0.0
    return float(np.abs(current - target).sum() / total)


def _solve_joint_weights(
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
    target_bundle: list[dict[str, object]],
    *,
    household_total: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    improvement_tol: float = DEFAULT_IMPROVEMENT_TOL,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Run a simple household-donor joint IPU-style reweighting loop."""
    base_weights = pd.to_numeric(household_donors["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if base_weights.sum() <= 0:
        base_weights = np.ones(len(household_donors), dtype=float)
    weights = base_weights / base_weights.sum() * max(float(household_total), 1.0)

    history: list[dict[str, object]] = []
    prev_objective: float | None = None
    for iteration in range(1, max_iterations + 1):
        for bundle in target_bundle:
            matrix = bundle["matrix"]
            target = bundle["target_vector"]
            tier_step = 0.25 * float(bundle["tier_weight"])
            current = weights @ matrix
            ratios = np.divide(target, np.maximum(current, 1e-9))
            log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
            activity = matrix.sum(axis=1)
            factors = np.exp(np.where(activity > 0, (matrix @ log_ratios) / np.maximum(activity, 1.0), 0.0) * tier_step)
            weights *= factors
        if weights.sum() > 0:
            weights = weights / weights.sum() * max(float(household_total), 1.0)

        objective = 0.0
        tier_errors = {"tier_1": 0.0, "tier_2": 0.0, "tier_3": 0.0}
        for bundle in target_bundle:
            current = weights @ bundle["matrix"]
            error = _weighted_fit_error(current, bundle["target_vector"])
            bundle["fit_error"] = error
            objective += float(bundle["tier_weight"]) * error
            tier_errors[bundle["tier"]] = max(tier_errors[bundle["tier"]], error)

        history.append(
            {
                "iteration": iteration,
                "objective": float(objective),
                "tier_1_max_error": float(tier_errors["tier_1"]),
                "tier_2_max_error": float(tier_errors["tier_2"]),
                "tier_3_max_error": float(tier_errors["tier_3"]),
            }
        )
        improvement = prev_objective - objective if prev_objective is not None else None
        if improvement is not None and improvement >= 0:
            if improvement < improvement_tol and all(
                tier_errors[tier] <= threshold for tier, threshold in TIER_THRESHOLDS.items()
            ):
                break
        prev_objective = objective

    return weights, history


def _sample_integer_households(weights: np.ndarray, household_total: int) -> np.ndarray:
    """Convert fitted fractional household weights into integer donor draws."""
    counts = _largest_remainder_to_total({idx: float(value) for idx, value in enumerate(weights)}, household_total)
    return np.array(list(counts.values()), dtype=int)


def _candidate_household_pool(
    household_donors: pd.DataFrame,
    member_donors: pd.DataFrame,
    *,
    household_total: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a weighted candidate household pool for one DA-sized joint fit."""
    candidate_n = min(len(household_donors), max(120, int(household_total)))
    if candidate_n >= len(household_donors):
        return household_donors.copy(), member_donors.copy()
    weights = pd.to_numeric(household_donors["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(household_donors), dtype=float)
    probs = weights / weights.sum()
    chosen = np.sort(rng.choice(np.arange(len(household_donors)), size=int(candidate_n), replace=False, p=probs))
    candidate_households = household_donors.iloc[chosen].reset_index(drop=True)
    candidate_ids = set(candidate_households["household_id"].astype(str).tolist())
    candidate_members = member_donors.loc[member_donors["household_id"].astype(str).isin(candidate_ids)].copy().reset_index(drop=True)
    return candidate_households, candidate_members


def _build_household_coherence_audit(households_df: pd.DataFrame) -> pd.DataFrame:
    """Flag impossible household size/type combinations in synthesized outputs."""
    if households_df.empty:
        return pd.DataFrame(columns=["area", "household_id", "household_size", "household_type", "coherence_issue"])
    hh = households_df.copy()
    hh["household_size"] = hh["household_size"].astype(str)
    hh["household_type"] = hh["household_type"].astype(str)
    size_num = hh["household_size"].map({"1": 1, "2": 2, "3": 3, "4": 4, "5plus": 5})
    issues: list[pd.DataFrame] = []
    one_person = hh.loc[size_num.ne(1) & hh["household_type"].eq("one_person")].copy()
    if not one_person.empty:
        one_person["coherence_issue"] = "one_person_requires_size_1"
        issues.append(one_person)
    couple_wo = hh.loc[size_num.ne(2) & hh["household_type"].eq("couple_without_children")].copy()
    if not couple_wo.empty:
        couple_wo["coherence_issue"] = "couple_without_children_requires_size_2"
        issues.append(couple_wo)
    one_parent = hh.loc[size_num.lt(2) & hh["household_type"].eq("one_parent")].copy()
    if not one_parent.empty:
        one_parent["coherence_issue"] = "one_parent_requires_size_2plus"
        issues.append(one_parent)
    couple_w = hh.loc[size_num.lt(3) & hh["household_type"].eq("couple_with_children")].copy()
    if not couple_w.empty:
        couple_w["coherence_issue"] = "couple_with_children_requires_size_3plus"
        issues.append(couple_w)
    if not issues:
        return pd.DataFrame(columns=["area", "household_id", "household_size", "household_type", "coherence_issue"])
    return pd.concat(issues, ignore_index=True)[
        ["area", "household_id", "household_size", "household_type", "coherence_issue"]
    ].sort_values(["coherence_issue", "area", "household_id"]).reset_index(drop=True)


def _canonicalize_people_output(
    people_df: pd.DataFrame,
    *,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Project workflow person records onto the supported public bundle schema."""
    canonical_cols = [
        "area",
        "HID",
        "sex",
        "age_group",
        "education_level",
        "labour_force_status",
        "household_income",
        "family_status",
        "household_size",
        "household_type",
        "person_uid",
        "citizenship_status",
        "immigrant_status",
        "commute_mode",
        "commute_duration",
    ]
    keep = [col for col in canonical_cols if col in people_df.columns]
    out = people_df.loc[:, keep].copy()
    for column in out.columns:
        out[column] = public_value_series(
            column,
            out[column],
            age_group_scheme=age_group_scheme,
            age_group_breaks=age_group_breaks,
        )
    return out


def _canonicalize_households_output(households_df: pd.DataFrame) -> pd.DataFrame:
    """Project workflow household records onto the supported public bundle schema."""
    return households_df.copy()


def _aggregate_attribute_counts(
    df: pd.DataFrame,
    spec: JointControlSpec,
) -> dict[object, float]:
    """Aggregate synthetic output counts for one control using its eligible universe."""
    if spec.unit == "household":
        series = df[spec.attribute]
        return {category: float(series.eq(category).sum()) for category in spec.categories}
    eligible = _eligible_members(df, spec.restricted_universe)
    series = eligible[spec.attribute]
    return {category: float(series.eq(category).sum()) for category in spec.categories}


def _summarize_joint_fit_outputs(
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    target_bundle: list[dict[str, object]],
    age_group_scheme: str,
    age_group_breaks: list[int] | tuple[int, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare synthesized outputs against the unified smoothed targets."""
    summary_columns = [
        "unit",
        "attribute",
        "control_tier",
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
        "target_provenance",
        "smoothing_mode",
        "raw_target_total",
        "effective_target_total",
        "donor_support_mass",
        "fit_error",
    ]
    detail_columns = [
        "unit",
        "attribute",
        "category",
        "reference_count",
        "seed_count",
        "reference_share",
        "seed_share",
        "abs_pp_diff",
        "target_provenance",
        "smoothing_mode",
    ]
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for bundle in target_bundle:
        spec: JointControlSpec = bundle["spec"]
        effective_counts: dict[object, int] = bundle["effective_counts"]
        actual_counts = _aggregate_attribute_counts(people_df if spec.unit == "person" else households_df, spec)
        reference_total = float(sum(effective_counts.values()))
        seed_total = float(sum(actual_counts.values()))
        categories = list(spec.categories)
        reference = pd.Series({category: float(effective_counts.get(category, 0.0)) for category in categories}, dtype=float)
        actual = pd.Series({category: float(actual_counts.get(category, 0.0)) for category in categories}, dtype=float)
        if reference_total > 0:
            ref_share = reference / reference_total
            act_share = actual / max(seed_total, 1.0)
            diff = act_share - ref_share
            tvd = float(0.5 * np.abs(diff).sum())
            mae_pp = float(np.abs(diff).mean() * 100.0)
            rmse_pp = float(np.sqrt((diff.pow(2)).mean()) * 100.0)
            max_abs_pp = float(np.abs(diff).max() * 100.0)
        else:
            tvd = mae_pp = rmse_pp = max_abs_pp = np.nan

        summary_rows.append(
            {
                "unit": spec.unit,
                "attribute": public_attr_name(spec.attribute),
                "control_tier": spec.tier,
                "workflow_role": "joint_fit",
                "reference_available": True,
                "reference_source": JOINT_REFERENCE_SOURCE,
                "reference_total": reference_total,
                "census_available": True,
                "census_source": JOINT_REFERENCE_SOURCE,
                "census_total": reference_total,
                "seed_total": seed_total,
                "seed_to_reference_ratio": (seed_total / reference_total) if reference_total > 0 else np.nan,
                "seed_to_census_ratio": (seed_total / reference_total) if reference_total > 0 else np.nan,
                "n_categories": int(len(categories)),
                "tvd": tvd,
                "mae_pp": mae_pp,
                "rmse_pp": rmse_pp,
                "max_abs_pp": max_abs_pp,
                "target_provenance": bundle["target_provenance"],
                "smoothing_mode": bundle["smoothing_mode"],
                "raw_target_total": float(bundle["raw_target_total"]),
                "effective_target_total": reference_total,
                "donor_support_mass": float(bundle["donor_support_mass"]),
                "fit_error": float(bundle["fit_error"]),
            }
        )
        for category in categories:
            display_category = public_value_label(spec.attribute, category, age_group_scheme=age_group_scheme, age_group_breaks=age_group_breaks)
            detail_rows.append(
                {
                    "unit": spec.unit,
                    "attribute": public_attr_name(spec.attribute),
                    "category": display_category,
                    "reference_count": float(reference.loc[category]),
                    "seed_count": float(actual.loc[category]),
                    "reference_share": float(reference.loc[category] / reference_total) if reference_total > 0 else np.nan,
                    "seed_share": float(actual.loc[category] / seed_total) if seed_total > 0 else np.nan,
                    "abs_pp_diff": (
                        float(abs((actual.loc[category] / seed_total) - (reference.loc[category] / reference_total)) * 100.0)
                        if reference_total > 0 and seed_total > 0
                        else np.nan
                    ),
                    "target_provenance": bundle["target_provenance"],
                    "smoothing_mode": bundle["smoothing_mode"],
                }
            )
    return pd.DataFrame(summary_rows, columns=summary_columns), pd.DataFrame(detail_rows, columns=detail_columns)


def build_joint_workflow_plan_artifacts(
    *,
    context_tables: dict[str, pd.DataFrame],
    da_codes: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create transitional planning artifacts for the joint method."""
    support_rows = []
    for spec in JOINT_CONTROL_SPECS:
        support_rows.append(
            {
                "attribute": public_attr_name(spec.attribute),
                "unit": spec.unit,
                "workflow_role": "joint_fit",
                "assignment_route": "joint_fit",
                "support_class": "joint_fit",
                "control_tier": spec.tier,
                "target_source": spec.target_table or "seed_prior",
                "target_provenance": "planned",
                "smoothing_mode": "borrow_and_smooth",
                "effective_target_total": pd.NA,
                "raw_target_total": pd.NA,
                "fit_error": pd.NA,
                "min_category_weight": pd.NA,
                "min_conditional_weight": pd.NA,
            }
        )
    workflow_plan = {
        "workflow_method": "joint_ipu_v1",
        "selected_da_count": int(len(da_codes)),
        "selected_da_codes": da_codes,
        "derived_output_attributes": list(DERIVED_OUTPUT_ATTRIBUTES),
        "workflow_steps": [
            "prepare_inputs_and_scope",
            "build_unified_controls",
            "smooth_da_targets",
            "fit_joint_household_donors",
            "materialize_synthetic_population",
            "validate_joint_fit",
        ],
        "all_workflow_attributes": [
            {
                "attribute": public_attr_name(spec.attribute),
                "unit": spec.unit,
                "workflow_role": "joint_fit",
                "control_tier": spec.tier,
                "target_source": spec.target_table or "seed_prior",
                "restricted_universe": spec.restricted_universe,
            }
            for spec in JOINT_CONTROL_SPECS
        ],
    }
    return pd.DataFrame(support_rows), workflow_plan


def run_joint_ipu_workflow(
    *,
    data_root: str | Path,
    census_pumf_root: str | Path,
    housing_survey_root: str | Path | None,
    base_population_path: str | Path | None,
    output_dir: str | Path,
    province: str = "24",
    geography_scope: str = "montreal",
    random_seed: int = 42,
    max_das: int | None = None,
    da_codes: list[str] | None = None,
    da_scope_name: str | None = None,
    da_codes_file: str | Path | None = None,
    processed_inputs_dir: str | Path | None = None,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
    show_progress: bool = True,
) -> JointWorkflowArtifacts:
    """Execute the unified joint-fit workflow for a DA selection."""
    del housing_survey_root, base_population_path, show_progress
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if processed_inputs_dir is not None:
        processed = load_processed_artifacts(processed_inputs_dir)
        context_tables = load_processed_context_tables(processed)
    else:
        context_tables = load_context_tables(data_root)
    explicit_da_codes = resolve_da_scope_codes(
        da_codes=da_codes,
        da_scope_name=da_scope_name,
        da_codes_file=da_codes_file,
    )
    if explicit_da_codes is not None:
        selected_da_codes = list(explicit_da_codes)
    else:
        all_codes = sorted(
            {str(code).strip() for table in context_tables.values() if "da_code" in table.columns for code in table["da_code"].astype(str)}
        )
        selected_da_codes = all_codes[: int(max_das)] if max_das is not None else all_codes
    household_donors, member_donors = _prepare_joint_household_donors(census_pumf_root, province=province)

    support_df, workflow_plan = build_joint_workflow_plan_artifacts(
        context_tables=context_tables,
        da_codes=selected_da_codes,
    )
    workflow_plan_path = out_dir / "energy_workflow_plan.json"
    workflow_plan_path.write_text(json.dumps(workflow_plan, indent=2), encoding="utf-8")

    people_rows: list[dict[str, object]] = []
    household_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    route_rows: list[dict[str, object]] = []
    sparse_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    all_target_bundles: list[dict[str, object]] = []
    household_counter = 0
    rng = np.random.default_rng(int(random_seed))
    donor_groups = {str(hh_id): frame.copy() for hh_id, frame in member_donors.groupby("household_id", sort=False)}

    for da_index, da_code in enumerate(selected_da_codes):
        household_total = _household_total_from_context(context_tables, da_code)
        if household_total <= 0:
            continue
        candidate_households, candidate_members = _candidate_household_pool(
            household_donors,
            member_donors,
            household_total=household_total,
            rng=np.random.default_rng(int(random_seed) + da_index),
        )
        citizenship_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "citizenship_status")
        person_total_hint = int(
            sum(
                _extract_direct_counts(
                    citizenship_spec,
                    _get_da_row(context_tables.get("immigration_citizenship", pd.DataFrame()), da_code),
                ).values()
            )
        )
        if person_total_hint <= 0:
            weighted_sizes = pd.to_numeric(household_donors["household_size"].replace({"5plus": "5"}), errors="coerce").fillna(1.0)
            donor_weights = pd.to_numeric(household_donors["weight"], errors="coerce").fillna(1.0)
            person_total_hint = int(round(float(np.average(weighted_sizes, weights=donor_weights)) * float(household_total)))

        target_bundle: list[dict[str, object]] = []
        for spec in JOINT_CONTROL_SPECS:
            effective_counts, metadata = _smoothed_target_for_da(
                spec,
                da_code=da_code,
                da_codes=selected_da_codes,
                context_tables=context_tables,
                household_donors=candidate_households,
                member_donors=candidate_members,
                household_total=household_total,
                person_total_hint=person_total_hint,
            )
            matrix = _household_incidence_matrix(spec, candidate_households, candidate_members)
            target_vector = np.array([float(effective_counts.get(category, 0.0)) for category in spec.categories], dtype=float)
            donor_support_mass = float((matrix.sum(axis=1) > 0).sum())
            target_bundle.append(
                {
                    "spec": spec,
                    "matrix": matrix,
                    "target_vector": target_vector,
                    "effective_counts": effective_counts,
                    "tier": spec.tier,
                    "tier_weight": spec.tier_weight,
                    "target_provenance": metadata["target_provenance"],
                    "smoothing_mode": metadata["smoothing_mode"],
                    "raw_target_total": metadata["raw_target_total"],
                    "effective_target_total": metadata["effective_target_total"],
                    "donor_support_mass": donor_support_mass,
                    "fit_error": np.nan,
                    "area": str(da_code),
                }
            )

        fitted_weights, history = _solve_joint_weights(
            candidate_households,
            candidate_members,
            target_bundle,
            household_total=household_total,
        )
        for row in history:
            history_rows.append({"area": str(da_code), **row})
        integer_counts = _sample_integer_households(fitted_weights, household_total)
        for bundle in target_bundle:
            support_rows.append(
                {
                    "attribute": public_attr_name(bundle["spec"].attribute),
                    "unit": bundle["spec"].unit,
                    "workflow_role": "joint_fit",
                    "assignment_route": "joint_fit",
                    "support_class": bundle["target_provenance"],
                    "control_tier": bundle["spec"].tier,
                    "target_source": bundle["spec"].target_table or "seed_prior",
                    "target_provenance": bundle["target_provenance"],
                    "smoothing_mode": bundle["smoothing_mode"],
                    "raw_target_total": bundle["raw_target_total"],
                    "effective_target_total": bundle["effective_target_total"],
                    "donor_support_mass": bundle["donor_support_mass"],
                    "fit_error": float(bundle["fit_error"]),
                    "min_category_weight": bundle["donor_support_mass"],
                    "min_conditional_weight": bundle["donor_support_mass"],
                }
            )
            route_rows.append(
                {
                    "area": str(da_code),
                    "unit": bundle["spec"].unit,
                    "attribute": public_attr_name(bundle["spec"].attribute),
                    "planned_route": "joint_fit",
                    "selected_route": "joint_fit",
                    "recommended_conditioning_json": json.dumps([]),
                    "planned_min_conditional_weight": float(bundle["donor_support_mass"]),
                    "observed_min_conditional_weight": float(bundle["donor_support_mass"]),
                    "downgraded_to_sparse": False,
                }
            )
            all_target_bundles.append(bundle)

        donor_indices = [idx for idx, count in enumerate(integer_counts.tolist()) for _ in range(int(count))]
        rng.shuffle(donor_indices)
        for donor_idx in donor_indices:
            donor_household = candidate_households.iloc[int(donor_idx)]
            donor_members = donor_groups.get(str(donor_household["household_id"]))
            if donor_members is None or donor_members.empty:
                continue
            household_counter += 1
            household_key = str(household_counter)
            household_rows.append(
                {
                    "area": str(da_code),
                    "household_id": household_key,
                    "household_size": donor_household["household_size"],
                    "household_type": donor_household["household_type"],
                    "bedrooms": donor_household["bedrooms"],
                    "core_housing_need": donor_household["core_housing_need"],
                    "dwelling_condition": donor_household["dwelling_condition"],
                    "dwelling_type": donor_household["dwelling_type"],
                    "period_built": donor_household["period_built"],
                    "tenure": donor_household["tenure"],
                }
            )
            for _, member in donor_members.iterrows():
                people_rows.append(
                    {
                        "area": str(da_code),
                        "HID": household_key,
                        "sex": member["sex"],
                        "age_group": member["age_group"],
                        "education_level": member["education_level"],
                        "labour_force_status": member["labour_force_status"],
                        "household_income": member["household_income"],
                        "family_status": member["family_status"],
                        "household_size": donor_household["household_size"],
                        "household_type": donor_household["household_type"],
                        "citizenship_status": member["citizenship_status"],
                        "immigrant_status": member["immigrant_status"],
                        "commute_mode": member["commute_mode"],
                        "commute_duration": member["commute_duration"],
                    }
                )

    people_out = pd.DataFrame(people_rows)
    households_out = pd.DataFrame(household_rows)
    people_out["person_uid"] = np.arange(len(people_out))

    household_coherence_df = _build_household_coherence_audit(households_out)
    summary_df, details_df = _summarize_joint_fit_outputs(
        people_df=people_out,
        households_df=households_out,
        target_bundle=all_target_bundles,
        age_group_scheme=age_group_scheme,
        age_group_breaks=age_group_breaks,
    )
    results_summary_df = build_results_summary(summary_df)
    support_columns = [
        "attribute",
        "unit",
        "workflow_role",
        "assignment_route",
        "support_class",
        "control_tier",
        "target_source",
        "target_provenance",
        "smoothing_mode",
        "raw_target_total",
        "effective_target_total",
        "donor_support_mass",
        "fit_error",
        "min_category_weight",
        "min_conditional_weight",
    ]
    route_columns = [
        "area",
        "unit",
        "attribute",
        "planned_route",
        "selected_route",
        "recommended_conditioning_json",
        "planned_min_conditional_weight",
        "observed_min_conditional_weight",
        "downgraded_to_sparse",
    ]
    sparse_columns = [
        "area",
        "unit",
        "attribute",
        "support_class",
        "fallback_rank",
        "chosen_conditioning_json",
        "used_global_fallback",
    ]
    support_report_df = pd.DataFrame(support_rows, columns=support_columns)
    assignment_route_df = pd.DataFrame(route_rows, columns=route_columns)
    sparse_report_df = pd.DataFrame(sparse_rows, columns=sparse_columns)

    people_public = _canonicalize_people_output(
        people_out,
        age_group_scheme=age_group_scheme,
        age_group_breaks=age_group_breaks,
    )
    households_public = _canonicalize_households_output(households_out)

    people_path = out_dir / "energy_workflow_people.parquet"
    households_path = out_dir / "energy_workflow_households.parquet"
    people_public.to_parquet(people_path, index=False)
    households_public.to_parquet(households_path, index=False)

    summary_path = out_dir / "energy_workflow_metric_summary.csv"
    details_path = out_dir / "energy_workflow_metric_details.csv"
    support_path = out_dir / "energy_workflow_support_classification.csv"
    sparse_path = out_dir / "energy_workflow_sparse_handling.csv"
    route_path = out_dir / "energy_workflow_assignment_route_decisions.csv"
    coherence_path = out_dir / "energy_workflow_household_coherence_audit.csv"
    results_summary_path = out_dir / "energy_workflow_results_summary.csv"
    convergence_path = out_dir / "energy_workflow_joint_convergence.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)
    support_report_df.to_csv(support_path, index=False)
    sparse_report_df.to_csv(sparse_path, index=False)
    assignment_route_df.to_csv(route_path, index=False)
    household_coherence_df.to_csv(coherence_path, index=False)
    results_summary_df.to_csv(results_summary_path, index=False)
    pd.DataFrame(history_rows).to_csv(convergence_path, index=False)

    metadata = {
        "method": "joint_ipu_v1",
        "reference_source": JOINT_REFERENCE_SOURCE,
        "province": province,
        "geography_scope": geography_scope,
        "selected_da_count": int(len(selected_da_codes)),
        "selected_da_codes": selected_da_codes,
        "n_people": int(len(people_out)),
        "n_households": int(len(households_out)),
        "workflow_steps": workflow_plan["workflow_steps"],
        "all_workflow_attributes": [public_attr_name(spec.attribute) for spec in JOINT_CONTROL_SPECS],
        "derived_output_attributes": list(DERIVED_OUTPUT_ATTRIBUTES),
        "control_tiers": {public_attr_name(spec.attribute): spec.tier for spec in JOINT_CONTROL_SPECS},
        "convergence_csv": str(convergence_path),
    }
    metadata_path = out_dir / "energy_workflow_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return JointWorkflowArtifacts(
        output_dir=out_dir,
        people_parquet=people_path,
        households_parquet=households_path,
        workflow_plan_json=workflow_plan_path,
        assignment_route_csv=route_path,
        household_coherence_csv=coherence_path,
        summary_csv=summary_path,
        details_csv=details_path,
        results_summary_csv=results_summary_path,
        support_classification_csv=support_path,
        sparse_handling_csv=sparse_path,
        metadata_json=metadata_path,
    )
