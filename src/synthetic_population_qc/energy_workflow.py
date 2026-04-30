"""Official end-to-end workflow for bundle-first synthetic population generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import humanleague
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from synthetic_population_qc.context_tables import load_context_tables
from synthetic_population_qc.config import default_geometry_dir
from synthetic_population_qc.enrichment import HOUSEHOLD_ATTR_LABELS, PERSON_ATTR_LABELS
from synthetic_population_qc.ingest.preprocess import load_processed_context_tables
from synthetic_population_qc.reporting import build_results_summary
from synthetic_population_qc.seed_preparation import (
    _derive_household_type_from_members,
    _prepare_core_seed,
    export_prepared_seed_artifacts,
    load_census_hierarchical_pumf,
)
from synthetic_population_qc.public_schema import public_attr_name, public_conditioning_cols, public_value_label, public_value_series
from synthetic_population_qc.runs.bundle import load_processed_artifacts
from synthetic_population_qc.scope_selection import resolve_da_scope_codes
from synthetic_population_qc.seed_transforms import probabilistic_sampling
from synthetic_population_qc.sparse_handling import assign_attribute_with_fallback, conditional_support_weight
from synthetic_population_qc.support_assessment import SUPPORT_SPECS, build_support_assessment, support_strategy_map
from synthetic_population_qc.workflow_inputs import find_household_pumf_candidates


HOUSEHOLD_COLLAPSE_MAP = {
    "dwelling_type": {
        "single_detached_house": ["single_detached"],
        "apartment": ["duplex_apartment", "lowrise_apartment", "highrise_apartment"],
        "other_dwelling": ["semi_detached", "row_house", "other_single_attached", "movable_dwelling"],
    },
    "tenure": {
        "owner": ["owner"],
        "renter_or_band": ["renter", "band_housing"],
    },
    "bedrooms": {
        "no_bedroom": ["no_bedrooms"],
        "1_bedroom": ["1_bedroom"],
        "2_bedrooms": ["2_bedrooms"],
        "3_bedrooms": ["3_bedrooms"],
        "4plus_bedrooms": ["4plus_bedrooms"],
    },
    "period_built": {
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
    },
    "dwelling_condition": {
        "regular_maintenance": ["regular_or_minor_repairs"],
        "minor_repairs": ["regular_or_minor_repairs"],
        "major_repairs": ["major_repairs_needed"],
    },
    "core_housing_need": {
        "not_in_core_need": ["not_in_core_need"],
        "in_core_need": ["in_core_need"],
    },
}


PERSON_COLLAPSE_MAP = {
    "citizenship_status": {
        "canadian_citizen": ["canadian_citizen"],
        "not_canadian_citizen": ["not_canadian_citizen"],
    },
    "immigrant_status": {
        "non_immigrant": ["non_immigrant"],
        "immigrant": ["immigrant"],
        "non_permanent_resident": ["non_permanent_resident"],
    },
    "commute_mode": {
        "private_vehicle": ["car_truck_van"],
        "public_transit": ["public_transit"],
        "active_transport": ["walked", "bicycle"],
        "other_method": ["other_method"],
    },
    "commute_duration": {
        "lt_15_min": ["lt_15_min"],
        "15_to_29_min": ["15_to_29_min"],
        "30_to_44_min": ["30_to_44_min"],
        "45_to_59_min": ["45_to_59_min"],
        "60_plus_min": ["60plus_min"],
    },
}


HOUSEHOLD_TYPE_LABELS = {
    0: "couple_without_children",
    1: "couple_with_children",
    2: "one_parent",
    3: "one_person",
    4: "other",
}


@dataclass(frozen=True)
class EnergyWorkflowArtifacts:
    """Typed file outputs produced by one workflow execution."""
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


def _load_base_population(base_population_path: str | Path) -> pd.DataFrame:
    """Load a cached base population and normalize its DA code column."""
    df = pd.read_parquet(base_population_path)
    out = df.copy()
    out["area"] = out["area"].astype(str).str.strip()
    return out


def _with_alias_columns(df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    """Backfill missing columns from known aliases without overwriting data."""
    out = df.copy()
    for target, source in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def _ensure_semantic_person_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize person-side records onto the workflow's semantic schema."""
    out = df.copy()
    sex_numeric = pd.to_numeric(out["sex"], errors="coerce") if "sex" in out.columns else pd.Series(index=out.index, dtype=float)
    if "sex" not in out.columns or sex_numeric.notna().any():
        if "sex_native" in out.columns:
            out["sex"] = out["sex_native"]
        else:
            out["sex"] = sex_numeric.map({0: "female", 1: "male"}).where(sex_numeric.notna(), out.get("sex"))
    alias_map = {
        "age_group": "agegrp_core",
        "education_level": "hdgree_core",
        "labour_force_status": "lfact_core",
        "household_income": "totinc_core",
        "family_status": "cfstat_core",
    }
    out = _with_alias_columns(out, alias_map)
    if "household_size" not in out.columns:
        if "household_size_native" in out.columns:
            out["household_size"] = out["household_size_native"]
        elif "hhsize_core" in out.columns:
            out["household_size"] = pd.to_numeric(out["hhsize_core"], errors="coerce").map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"})
    if "household_type" not in out.columns:
        if "household_type_native" in out.columns:
            out["household_type"] = out["household_type_native"]
        elif "hhtype" in out.columns:
            out["household_type"] = pd.to_numeric(out["hhtype"], errors="coerce").map(HOUSEHOLD_TYPE_LABELS)
    return out


def _ensure_semantic_household_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize household-side records onto the workflow's semantic schema."""
    out = df.copy()
    out = _with_alias_columns(
        out,
        {
            "household_size": "household_size_native",
            "household_type": "household_type_native",
        },
    )
    return out


def _prepare_hierarchical_member_seed(
    census_pumf_root: str | Path,
    *,
    province: str = "24",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a member-level donor seed used for base-population generation."""
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_parquet(cache_path)
    candidates = find_household_pumf_candidates(census_pumf_root)
    if not candidates:
        raise FileNotFoundError(f"No hierarchical census PUMF candidates found under {census_pumf_root}")
    raw = load_census_hierarchical_pumf(candidates[0], province=province)
    raw = raw.copy()
    raw["HHSIZE"] = raw.groupby("HH_ID", sort=False)["HH_ID"].transform("size")
    core = _prepare_core_seed(raw, gender_col="GENDER", totinc_col="TOTINC")
    hh_types = _derive_household_type_from_members(raw)
    out = core.merge(hh_types, on="HH_ID", how="left")
    out["household_size"] = (
        pd.to_numeric(out["n_members"], errors="coerce").clip(lower=1).fillna(1).astype(int).clip(upper=5).map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"})
    )
    out["sex_label"] = out["Gender"].map({1: "male", 2: "female"})
    out["weight"] = pd.to_numeric(out["WEIGHT"], errors="coerce").fillna(0.0)
    out["age_group"] = pd.to_numeric(out["agegrp"], errors="coerce")
    out["education_level"] = pd.to_numeric(out["hdgree"], errors="coerce")
    out["labour_force_status"] = pd.to_numeric(out["lfact"], errors="coerce")
    out["household_income"] = pd.to_numeric(out["TotInc"], errors="coerce")
    out["family_status"] = pd.to_numeric(out["cfstat"], errors="coerce")
    result = out[
        [
            "HH_ID",
            "PP_ID",
            "weight",
            "household_type",
            "household_size",
            "sex_label",
            "age_group",
            "education_level",
            "labour_force_status",
            "household_income",
            "family_status",
        ]
    ].reset_index(drop=True)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def _named_scope_da_codes_or_all(*, geography_scope: str, data_root: str | Path, context_tables: dict[str, pd.DataFrame]) -> list[str]:
    """Resolve a named geography into the DA codes available in context tables."""
    def _codes_from_table(table_name: str) -> set[str]:
        table = context_tables.get(table_name, pd.DataFrame())
        if table.empty or "da_code" not in table.columns:
            return set()
        return {str(x).strip() for x in table["da_code"].astype(str).tolist() if str(x).strip() and len(str(x).strip()) == 8}

    codes: set[str] = set()
    if str(geography_scope) != "all_qc":
        try:
            from urban_energy_core import load_city_da_geojsons

            gdf = load_city_da_geojsons(geometry_dir=default_geometry_dir(), show_progress=False).get(str(geography_scope))
            if gdf is not None and "DAUID" in gdf.columns:
                codes = set(gdf["DAUID"].astype(str).str.strip())
        except Exception:
            codes = set()
    # The geometry helper is unreliable in this environment for Montreal. Use
    # Montreal-specific DA tables as the official scope when available.
    if str(geography_scope) == "montreal":
        commute_codes = _codes_from_table("commute")
        household_type_codes = _codes_from_table("household_type_size_detailed")
        preferred_codes = commute_codes & household_type_codes if commute_codes and household_type_codes else (commute_codes or household_type_codes)
        if preferred_codes:
            codes = preferred_codes
    available_codes: set[str] = set()
    for key in ["housing", "dwelling_characteristics", "immigration_citizenship"]:
        table = context_tables.get(key, pd.DataFrame())
        if not table.empty and "da_code" in table.columns:
            available_codes |= set(table["da_code"].astype(str).tolist())
    available_codes = {str(x).strip() for x in available_codes if str(x).strip() and len(str(x).strip()) > 4}
    if codes:
        codes = codes & available_codes if available_codes else codes
    if not codes:
        codes = available_codes
    codes = {str(x).strip() for x in codes if str(x).strip() and len(str(x).strip()) > 4}
    return sorted(codes)


def _household_total_from_context(context_tables: dict[str, pd.DataFrame], da_code: str) -> int:
    """Estimate one DA's occupied-private-dwelling total from available context tables."""
    dwelling_row = _get_da_row(context_tables.get("dwelling_characteristics", pd.DataFrame()), da_code)
    if dwelling_row is not None:
        total = pd.to_numeric(
            pd.Series(
                [
                    dwelling_row.get(
                        "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data"
                    )
                ]
            ),
            errors="coerce",
        ).fillna(0.0).iloc[0]
        if total > 0:
            return int(round(float(total)))
    housing_row = _get_da_row(context_tables.get("housing", pd.DataFrame()), da_code)
    if housing_row is not None:
        total = 0.0
        for col in HOUSEHOLD_ATTR_LABELS["tenure"].values():
            total += float(pd.to_numeric(pd.Series([housing_row.get(col)]), errors="coerce").fillna(0.0).iloc[0])
        if total > 0:
            return int(round(total))
    return 0


def _build_household_type_targets_from_context(
    context_tables: dict[str, pd.DataFrame],
    da_code: str,
    *,
    household_seed: pd.DataFrame,
) -> dict[str, int]:
    """Build household-type targets for one DA from context or household seed shares."""
    table = context_tables.get("household_type_size_detailed", pd.DataFrame())
    row = _get_da_row(table, da_code)
    if row is not None:
        return {
            "couple_with_children": int(round(float(pd.to_numeric(pd.Series([row.get('Households by type / Total - Household type - 100% data / One-census-family households without additional persons / Couple-family households / With children')]), errors='coerce').fillna(0.0).iloc[0]))),
            "couple_without_children": int(round(float(pd.to_numeric(pd.Series([row.get('Households by type / Total - Household type - 100% data / One-census-family households without additional persons / Couple-family households / Without children')]), errors='coerce').fillna(0.0).iloc[0]))),
            "one_parent": int(round(float(pd.to_numeric(pd.Series([row.get('Households by type / Total - Household type - 100% data / One-census-family households without additional persons / One-parent-family households')]), errors='coerce').fillna(0.0).iloc[0]))),
            "one_person": int(round(float(pd.to_numeric(pd.Series([row.get('Households by type / Total - Household type - 100% data / One-person households')]), errors='coerce').fillna(0.0).iloc[0]))),
            "other": int(round(float(pd.to_numeric(pd.Series([row.get('Households by type / Total - Household type - 100% data / Two-or-more-person non-census-family households')]), errors='coerce').fillna(0.0).iloc[0]))),
        }
    total_hh = _household_total_from_context(context_tables, da_code)
    if total_hh <= 0:
        return {"couple_with_children": 0, "couple_without_children": 0, "one_parent": 0, "one_person": 0, "other": 0}
    seed_counts = household_seed["household_type"].value_counts().to_dict()
    return _largest_remainder_to_total({k: float(seed_counts.get(k, 0.0)) for k in HOUSEHOLD_TYPE_LABELS.values()}, total_hh)


def _sample_household_ids(
    seed_households: pd.DataFrame,
    *,
    household_type: str,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample household seed IDs with replacement for a requested household type."""
    pool = seed_households.loc[seed_households["household_type"] == household_type].copy()
    if pool.empty:
        pool = seed_households.copy()
    weights = pd.to_numeric(pool["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(pool), dtype=float)
    probs = weights / weights.sum()
    chosen = rng.choice(pool["household_id"].astype(str).to_numpy(), size=int(n), replace=True, p=probs)
    return chosen.tolist()


def _selected_da_codes(
    *,
    geography_scope: str,
    data_root: str | Path,
    context_tables: dict[str, pd.DataFrame],
    max_das: int | None,
) -> list[str]:
    """Return the DA list to synthesize when no explicit DA list is provided."""
    da_codes = _named_scope_da_codes_or_all(
        geography_scope=geography_scope,
        data_root=data_root,
        context_tables=context_tables,
    )
    if max_das is not None:
        da_codes = da_codes[: int(max_das)]
    return da_codes


def _core_population_from_raw_inputs(
    *,
    data_root: str | Path,
    census_pumf_root: str | Path | None,
    output_dir: str | Path,
    province: str = "24",
    geography_scope: str = "montreal",
    random_seed: int = 42,
    max_das: int | None = None,
    da_codes: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate the base synthetic population directly from raw Canadian inputs."""
    core_dir = Path(output_dir)
    core_dir.mkdir(parents=True, exist_ok=True)
    if census_pumf_root is None:
        raise ValueError("census_pumf_root is required for end-to-end core population generation")
    context_tables = load_context_tables(data_root)
    da_codes = list(da_codes) if da_codes is not None else _selected_da_codes(
        geography_scope=geography_scope,
        data_root=data_root,
        context_tables=context_tables,
        max_das=max_das,
    )
    member_seed = _prepare_hierarchical_member_seed(
        census_pumf_root,
        province=province,
        cache_path=core_dir / f"hierarchical_member_seed_p{province}.parquet",
    )
    household_seed = (
        member_seed[["HH_ID", "household_type", "household_size", "weight"]]
        .drop_duplicates("HH_ID")
        .rename(columns={"HH_ID": "household_id"})
        .reset_index(drop=True)
    )
    member_groups = {str(hh_id): g.copy() for hh_id, g in member_seed.groupby("HH_ID", sort=False)}
    people_rows: list[dict[str, object]] = []
    hh_counter = 0
    iterator = tqdm(da_codes, desc="Core synthesis DAs") if show_progress else da_codes
    type_to_code = {v: k for k, v in HOUSEHOLD_TYPE_LABELS.items()}
    for da_code in iterator:
        type_counts = _build_household_type_targets_from_context(context_tables, da_code, household_seed=household_seed)
        total_hh = sum(type_counts.values())
        if total_hh <= 0:
            continue
        rng = np.random.default_rng(int(random_seed) + int(sum(ord(ch) for ch in str(da_code))))
        for household_type, n_households in type_counts.items():
            if int(n_households) <= 0:
                continue
            donor_ids = _sample_household_ids(household_seed, household_type=household_type, n=int(n_households), rng=rng)
            for donor_id in donor_ids:
                donor_members = member_groups.get(str(donor_id))
                if donor_members is None or donor_members.empty:
                    continue
                hh_counter += 1
                hhsize_val = int(
                    pd.Series([donor_members["household_size"].iloc[0]])
                    .astype(str)
                    .replace({"5plus": "5"})
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(1)
                    .clip(upper=5)
                    .iloc[0]
                )
                hhtype_val = int(type_to_code.get(household_type, 4))
                for _, member in donor_members.iterrows():
                    sex_code = 1 if str(member["sex_label"]).strip().lower() == "male" else 0
                    people_rows.append(
                        {
                            "area": str(da_code),
                            "HID": str(hh_counter),
                            "sex": sex_code,
                            "agegrp": member["age_group"],
                            "hdgree": member["education_level"],
                            "lfact": member["labour_force_status"],
                            "hhsize": hhsize_val,
                            "totinc": member["household_income"],
                            "cfstat": member["family_status"],
                            "hhtype": hhtype_val,
                        }
                    )
    base_people = pd.DataFrame(people_rows)
    if base_people.empty:
        raise RuntimeError("Core population synthesis completed without producing a base synthetic population.")
    base_people["area"] = base_people["area"].astype(str).str.strip()
    metadata = {
        "mode": "generated_from_raw_inputs",
        "geography_scope": geography_scope,
        "selected_da_count": int(len(da_codes)),
        "generation_method": "hierarchical_pumf_household_sampling",
    }
    pd.DataFrame({"da_code": da_codes}).to_csv(core_dir / "selected_das_core_population.csv", index=False)
    base_people.to_parquet(core_dir / "syn_inds_with_hh_core_population.parquet", index=False)
    return base_people, metadata


def _resolve_or_generate_base_population(
    *,
    data_root: str | Path,
    census_pumf_root: str | Path | None,
    output_dir: str | Path,
    base_population_path: str | Path | None,
    province: str = "24",
    geography_scope: str = "montreal",
    random_seed: int = 42,
    max_das: int | None = None,
    da_codes: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a prebuilt base population or generate one from raw inputs."""
    if base_population_path is not None:
        return _load_base_population(base_population_path), {
            "mode": "loaded_from_path",
            "base_population_path": str(Path(base_population_path)),
        }
    return _core_population_from_raw_inputs(
        data_root=data_root,
        census_pumf_root=census_pumf_root,
        output_dir=output_dir,
        province=province,
        geography_scope=geography_scope,
        random_seed=random_seed,
        max_das=max_das,
        da_codes=da_codes,
        show_progress=show_progress,
    )


def _derive_core_level_maps(person_seed_df: pd.DataFrame) -> dict[str, list]:
    """Capture the semantic person-level state space from the seed microdata."""
    return {
        "sex": ["female", "male"],
        "age_group": sorted(pd.to_numeric(person_seed_df["age_group"], errors="coerce").dropna().unique().tolist()),
        "education_level": sorted(pd.to_numeric(person_seed_df["education_level"], errors="coerce").dropna().unique().tolist()),
        "labour_force_status": sorted(pd.to_numeric(person_seed_df["labour_force_status"], errors="coerce").dropna().unique().tolist()),
        "household_income": sorted(pd.to_numeric(person_seed_df["household_income"], errors="coerce").dropna().unique().tolist()),
        "family_status": sorted(pd.to_numeric(person_seed_df["family_status"], errors="coerce").dropna().unique().tolist()),
    }


def _map_base_people_to_core_schema(base_people: pd.DataFrame, core_levels: dict[str, list]) -> pd.DataFrame:
    """Map raw coded base-population fields into the semantic workflow schema."""
    def _map_index_or_identity(series: pd.Series, levels: list) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        allowed_values = set(levels)
        identity = numeric.where(numeric.isin(allowed_values))
        mapped = numeric.map(dict(enumerate(levels)))
        return identity.where(identity.notna(), mapped.where(mapped.notna(), series.where(series.isin(allowed_values))))

    out = base_people.copy()
    sex_numeric = pd.to_numeric(out["sex"], errors="coerce")
    out["sex"] = sex_numeric.map({0: "female", 1: "male"})
    out["sex"] = out["sex"].where(out["sex"].notna(), out["sex"].where(out["sex"].isin({"female", "male"})))
    out["age_group"] = _map_index_or_identity(out["agegrp"], core_levels["age_group"])
    out["labour_force_status"] = _map_index_or_identity(out["lfact"], core_levels["labour_force_status"])
    out["education_level"] = _map_index_or_identity(out["hdgree"], core_levels["education_level"])
    out["household_income"] = _map_index_or_identity(out["totinc"], core_levels["household_income"])
    out["family_status"] = pd.to_numeric(out["cfstat"], errors="coerce").where(pd.to_numeric(out["cfstat"], errors="coerce").notna(), out["cfstat"])
    out["household_size"] = pd.to_numeric(out["hhsize"], errors="coerce").map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"})
    out["household_type"] = pd.to_numeric(out["hhtype"], errors="coerce").map(HOUSEHOLD_TYPE_LABELS)
    return out


def _build_base_households(base_people_native: pd.DataFrame) -> pd.DataFrame:
    """Collapse person records into one semantic row per household."""
    cols = ["area", "HID", "household_size", "household_type"]
    hh = (
        base_people_native.loc[base_people_native["HID"].notna(), cols]
        .drop_duplicates(["area", "HID"])
        .rename(columns={"HID": "household_id"})
        .reset_index(drop=True)
    )
    hh["household_id"] = hh["household_id"].astype(str)
    return hh


def _canonicalize_people_output(
    people_df: pd.DataFrame,
    *,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Project workflow people records onto the supported public bundle schema."""
    if people_df.empty:
        return people_df.copy()
    people_df = _ensure_semantic_person_schema(people_df)
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
    for col in out.columns:
        out[col] = public_value_series(
            col,
            out[col],
            age_group_scheme=age_group_scheme,
            age_group_breaks=age_group_breaks,
        )
    return out


def _canonicalize_households_output(households_df: pd.DataFrame) -> pd.DataFrame:
    """Project workflow household records onto the supported public bundle schema."""
    if households_df.empty:
        return households_df.copy()
    return _ensure_semantic_household_schema(households_df).copy()


def _build_workflow_plan(
    *,
    province: str,
    geography_scope: str,
    data_root: str | Path,
    base_population_path: str | Path | None,
    da_codes: list[str],
    support_df: pd.DataFrame,
) -> dict[str, object]:
    """Summarize the workflow steps for manifests and downstream tools."""
    plan_support = support_df.copy()
    if "workflow_role" not in plan_support.columns:
        plan_support["workflow_role"] = "extension"
    if "support_class" not in plan_support.columns:
        plan_support["support_class"] = pd.NA
    core_attrs = (
        plan_support.loc[plan_support["workflow_role"] == "core_generation", ["attribute", "unit", "assignment_route"]]
        .sort_values(["unit", "attribute"])
        .to_dict(orient="records")
    )
    stable_attrs = (
        plan_support.loc[
            plan_support["assignment_route"].isin(["direct_person_assignment", "direct_household_assignment"]),
            ["attribute", "unit", "assignment_route"],
        ]
        .sort_values(["unit", "attribute"])
        .to_dict(orient="records")
    )
    sparse_attrs = (
        plan_support.loc[plan_support["assignment_route"] == "sparse_fallback", ["attribute", "unit", "support_class"]]
        .sort_values(["unit", "attribute"])
        .to_dict(orient="records")
    )
    return {
        "province": province,
        "geography_scope": geography_scope,
        "data_root": str(Path(data_root)),
        "base_population_mode": "reuse_prebuilt" if base_population_path is not None else "generate_end_to_end",
        "workflow_steps": [
            "prepare_inputs_and_scope",
            "assess_support_and_plan",
            "generate_base_population",
            "assign_household_attributes",
            "assign_person_attributes",
            "apply_sparse_fallbacks",
            "evaluate_outputs",
        ],
        "selected_da_count": int(len(da_codes)),
        "selected_da_codes_preview": list(da_codes[:20]),
        "core_generation_attributes": [{**row, "attribute": public_attr_name(row["attribute"])} for row in core_attrs],
        "stable_attributes": [{**row, "attribute": public_attr_name(row["attribute"])} for row in stable_attrs],
        "sparse_attributes": [{**row, "attribute": public_attr_name(row["attribute"])} for row in sparse_attrs],
        "all_workflow_attributes": (
            [
                {**row, "attribute": public_attr_name(row["attribute"])}
                for row in plan_support[["attribute", "unit", "workflow_role", "assignment_route", "support_class"]]
                .sort_values(["unit", "workflow_role", "attribute"])
                .to_dict(orient="records")
            ]
        ),
    }


def _assignment_route_rows_for_da(
    *,
    da_code: str,
    unit: str,
    attrs: list[str],
    support_map: dict[str, dict[str, object]],
    seed_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Record per-DA assignment-route decisions for support-aware attributes."""
    rows: list[dict[str, object]] = []
    for attr in attrs:
        strat = support_map.get(attr, {})
        cond_cols = list(strat.get("recommended_cond_cols", []))
        planned_route = str(strat.get("assignment_route", "sparse_fallback"))
        planned_min = float(strat.get("min_conditional_weight", 0.0))
        observed_min = conditional_support_weight(seed_df, attr=attr, cond_cols=cond_cols)
        selected_route = planned_route if observed_min >= planned_min else "sparse_fallback"
        rows.append(
            {
                "area": str(da_code),
                "unit": unit,
                "attribute": attr,
                "public_attribute": public_attr_name(attr),
                "planned_route": planned_route,
                "selected_route": selected_route,
                "recommended_conditioning_json": json.dumps(public_conditioning_cols(cond_cols)),
                "planned_min_conditional_weight": planned_min,
                "observed_min_conditional_weight": float(observed_min),
                "downgraded_to_sparse": selected_route != planned_route,
            }
        )
    return rows


def _build_household_coherence_audit(households_df: pd.DataFrame) -> pd.DataFrame:
    """Flag impossible household size/type combinations in synthesized outputs."""
    if households_df.empty:
        return pd.DataFrame(
            columns=[
                "area",
                "household_id",
                "household_size",
                "household_type",
                "coherence_issue",
            ]
        )

    hh = _ensure_semantic_household_schema(households_df)
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


def _get_da_row(df: pd.DataFrame, da_code: str) -> pd.Series | None:
    """Return the first row matching one DA code from a context table."""
    if df.empty or "da_code" not in df.columns:
        return None
    match = df.loc[df["da_code"].astype(str).str.strip() == str(da_code).strip()]
    if match.empty:
        return None
    return match.iloc[0]


def _collapsed_counts_from_da_row(
    row: pd.Series | None,
    *,
    label_map: dict[str, str],
    collapse_map: dict[str, list[str]],
) -> dict[str, int]:
    """Collapse raw context-table columns into one harmonized target distribution."""
    counts: dict[str, int] = {}
    for out_key, in_keys in collapse_map.items():
        total = 0.0
        for in_key in in_keys:
            col = label_map.get(in_key)
            if col is None or row is None or col not in row.index:
                continue
            total += pd.to_numeric(pd.Series([row[col]]), errors="coerce").fillna(0.0).iloc[0]
        counts[out_key] = int(round(total))
    return counts


def _largest_remainder_to_total(targets: dict[str, float], total_n: int) -> dict[str, int]:
    """Integerize fractional targets while preserving the requested total."""
    if total_n <= 0:
        return {k: 0 for k in targets}
    vals = {k: max(0.0, float(v)) for k, v in targets.items()}
    current_total = sum(vals.values())
    if current_total > 0:
        vals = {k: (v / current_total) * float(total_n) for k, v in vals.items()}
    else:
        vals = {k: float(total_n) / float(len(vals)) for k in vals} if vals else {}
    floors = {k: int(np.floor(v)) for k, v in vals.items()}
    remainder = total_n - sum(floors.values())
    order = sorted(vals, key=lambda k: vals[k] - floors[k], reverse=True)
    for i in range(max(remainder, 0)):
        floors[order[i % len(order)]] += 1
    return floors


def _ensure_counts_sum(counts: dict[str, int], total_n: int) -> dict[str, int]:
    """Rebalance integer counts so they sum exactly to the requested total."""
    return _largest_remainder_to_total({k: float(v) for k, v in counts.items()}, total_n)


def _seed_array_from_df(seed_df: pd.DataFrame, dims: list[str], weight_col: str = "weight") -> tuple[np.ndarray, list[list]]:
    """Convert a weighted seed table into an IPF-ready dense array plus level lists."""
    work = seed_df[dims + [weight_col]].dropna(subset=dims).copy()
    levels = [sorted(work[col].astype(object).unique().tolist(), key=lambda x: str(x)) for col in dims]
    shape = [len(x) for x in levels]
    arr = np.ones(shape, dtype=float)
    code_maps = [{v: i for i, v in enumerate(level)} for level in levels]
    grouped = work.groupby(dims, dropna=False)[weight_col].sum().reset_index()
    for row in grouped.itertuples(index=False):
        idx = tuple(code_maps[i][getattr(row, dims[i])] for i in range(len(dims)))
        arr[idx] += float(getattr(row, weight_col))
    return arr, levels


def _target_vector(levels: list, counts: dict[str, int], total_n: int) -> np.ndarray:
    """Build one marginal target vector aligned to the seed levels."""
    vec = np.array([float(counts.get(level, 0)) for level in levels], dtype=float)
    if vec.sum() <= 0 and total_n > 0 and len(levels) > 0:
        vec[:] = float(total_n) / float(len(levels))
    return vec


def _synthesise_records_from_seed(
    *,
    seed_df: pd.DataFrame,
    dims: list[str],
    marginals: dict[str, dict[str, int]],
    total_n: int,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Run multiway IPF on a seed table and flatten the fitted result into records."""
    if total_n <= 0:
        return pd.DataFrame(columns=dims)
    seed_arr, levels = _seed_array_from_df(seed_df, dims, weight_col=weight_col)
    selectors = [np.array([i], dtype=int) for i in range(len(dims))]
    target_vectors = [_target_vector(levels[i], marginals[dims[i]], total_n) for i in range(len(dims))]
    fit = humanleague.ipf(seed_arr, selectors, target_vectors)
    fitted = fit["result"] if isinstance(fit, dict) else fit[0]
    fitted_int = probabilistic_sampling(fitted, total_n)
    flat = humanleague.flatten(fitted_int)
    out = pd.DataFrame()
    for idx, dim in enumerate(dims):
        out[dim] = [levels[idx][i] for i in flat[idx]]
    return out


def _assign_synth_values(
    *,
    base_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    match_cols: list[str],
    value_cols: list[str],
    da_col: str = "area",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign synthesized attribute values back onto a base record table by DA and match keys."""
    rng = np.random.default_rng(int(random_seed))
    base = base_df.copy().reset_index(drop=True)
    synth = synth_df.copy().reset_index(drop=True)
    base["_row_id"] = np.arange(len(base))
    synth["_srow_id"] = np.arange(len(synth))

    for frame in (base, synth):
        frame["_shuffle"] = rng.random(len(frame))
        frame.sort_values([da_col] + match_cols + ["_shuffle"], inplace=True)

    join_cols = [da_col] + match_cols
    base["_match_idx"] = base.groupby(join_cols, dropna=False).cumcount()
    synth["_match_idx"] = synth.groupby(join_cols, dropna=False).cumcount()
    matched = base.merge(
        synth[join_cols + ["_match_idx"] + value_cols + ["_srow_id"]],
        on=join_cols + ["_match_idx"],
        how="left",
    )

    remaining_base = matched.loc[matched["_srow_id"].isna(), base.columns].drop(columns=["_match_idx"])
    used_synth_ids = matched["_srow_id"].dropna().astype(int).unique().tolist()
    remaining_synth = synth.loc[~synth["_srow_id"].isin(used_synth_ids), [da_col] + value_cols + ["_srow_id"]].copy()

    if not remaining_base.empty and not remaining_synth.empty:
        remaining_base.sort_values([da_col, "_shuffle"], inplace=True)
        remaining_synth["_shuffle"] = rng.random(len(remaining_synth))
        remaining_synth.sort_values([da_col, "_shuffle"], inplace=True)
        remaining_base["_da_idx"] = remaining_base.groupby(da_col).cumcount()
        remaining_synth["_da_idx"] = remaining_synth.groupby(da_col).cumcount()
        fallback = remaining_base.merge(
            remaining_synth[[da_col, "_da_idx"] + value_cols + ["_srow_id"]],
            on=[da_col, "_da_idx"],
            how="left",
        )
        matched.loc[matched["_srow_id"].isna(), value_cols] = fallback[value_cols].to_numpy()

    out = matched.sort_values("_row_id").drop(columns=[c for c in matched.columns if c.startswith("_")])
    return out


def _build_household_targets(base_hh_da: pd.DataFrame, context_tables: dict[str, pd.DataFrame], da_code: str) -> dict[str, dict[str, int]]:
    """Build DA-specific household marginals on the semantic household schema."""
    n_households = int(len(base_hh_da))
    size_counts = _ensure_counts_sum(base_hh_da["household_size"].value_counts().to_dict(), n_households)
    type_counts = _ensure_counts_sum(base_hh_da["household_type"].value_counts().to_dict(), n_households)
    housing_row = _get_da_row(context_tables.get("housing", pd.DataFrame()), da_code)
    dwelling_row = _get_da_row(context_tables.get("dwelling_characteristics", pd.DataFrame()), da_code)
    return {
        "household_size": size_counts,
        "household_type": type_counts,
        "dwelling_type": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                dwelling_row, label_map=HOUSEHOLD_ATTR_LABELS["dwelling_type"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["dwelling_type"]
            ),
            n_households,
        ),
        "tenure": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                housing_row, label_map=HOUSEHOLD_ATTR_LABELS["tenure"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["tenure"]
            ),
            n_households,
        ),
        "bedrooms": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                housing_row, label_map=HOUSEHOLD_ATTR_LABELS["bedrooms"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["bedrooms"]
            ),
            n_households,
        ),
        "period_built": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                housing_row, label_map=HOUSEHOLD_ATTR_LABELS["period_built"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["period_built"]
            ),
            n_households,
        ),
        "dwelling_condition": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                housing_row, label_map=HOUSEHOLD_ATTR_LABELS["dwelling_condition"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["dwelling_condition"]
            ),
            n_households,
        ),
        "core_housing_need": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                housing_row, label_map=HOUSEHOLD_ATTR_LABELS["core_housing_need"], collapse_map=HOUSEHOLD_COLLAPSE_MAP["core_housing_need"]
            ),
            n_households,
        ),
    }


def _build_person_social_targets(base_people_da: pd.DataFrame, context_tables: dict[str, pd.DataFrame], da_code: str) -> dict[str, dict[str, int]]:
    """Build DA-specific person marginals for the stable person-synthesis stage."""
    n_people = int(len(base_people_da))
    imm_row = _get_da_row(context_tables.get("immigration_citizenship", pd.DataFrame()), da_code)
    return {
        "sex": _ensure_counts_sum(base_people_da["sex"].value_counts().to_dict(), n_people),
        "age_group": _ensure_counts_sum(base_people_da["age_group"].value_counts().to_dict(), n_people),
        "labour_force_status": _ensure_counts_sum(base_people_da["labour_force_status"].value_counts().to_dict(), n_people),
        "citizenship_status": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                imm_row, label_map=PERSON_ATTR_LABELS["citizenship_status"], collapse_map=PERSON_COLLAPSE_MAP["citizenship_status"]
            ),
            n_people,
        ),
        "immigrant_status": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                imm_row, label_map=PERSON_ATTR_LABELS["immigrant_status"], collapse_map=PERSON_COLLAPSE_MAP["immigrant_status"]
            ),
            n_people,
        ),
    }


def _build_person_mobility_targets(base_people_da: pd.DataFrame, context_tables: dict[str, pd.DataFrame], da_code: str) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Build DA-specific commute marginals for employed people only."""
    employed = base_people_da.loc[base_people_da["labour_force_status"] == 1867].copy()
    n_people = int(len(employed))
    commute_table = context_tables.get("commute", pd.DataFrame())
    commute_row = _get_da_row(commute_table, da_code)
    if commute_row is None and not commute_table.empty and "da_code" in commute_table.columns:
        aggregate_rows = commute_table.loc[commute_table["da_code"].astype(str).str.len() <= 4].copy()
        if not aggregate_rows.empty:
            commute_row = aggregate_rows.iloc[0]
    if n_people <= 0:
        return employed, {}
    return employed, {
        "sex": _ensure_counts_sum(employed["sex"].value_counts().to_dict(), n_people),
        "age_group": _ensure_counts_sum(employed["age_group"].value_counts().to_dict(), n_people),
        "commute_mode": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                commute_row, label_map=PERSON_ATTR_LABELS["commute_mode"], collapse_map=PERSON_COLLAPSE_MAP["commute_mode"]
            ),
            n_people,
        ),
        "commute_duration": _ensure_counts_sum(
            _collapsed_counts_from_da_row(
                commute_row, label_map=PERSON_ATTR_LABELS["commute_duration"], collapse_map=PERSON_COLLAPSE_MAP["commute_duration"]
            ),
            n_people,
        ),
    }


def _summarize_aligned_outputs(
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    *,
    reference_people_df: pd.DataFrame | None = None,
    reference_households_df: pd.DataFrame | None = None,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare synthesized outputs against workflow references and DA marginals."""
    summary_columns = [
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
    detail_columns = [
        "unit",
        "attribute",
        "category",
        "workflow_role",
        "reference_available",
        "reference_source",
        "reference_count",
        "census_available",
        "census_source",
        "census_count",
        "seed_count",
        "reference_share",
        "census_share",
        "seed_share",
        "abs_pp_diff",
    ]

    people_df = _ensure_semantic_person_schema(people_df)
    households_df = _ensure_semantic_household_schema(households_df)
    reference_people_df = _ensure_semantic_person_schema(reference_people_df) if reference_people_df is not None else pd.DataFrame()
    reference_households_df = _ensure_semantic_household_schema(reference_households_df) if reference_households_df is not None else pd.DataFrame()
    active_das = set()
    if "area" in people_df.columns:
        active_das |= set(people_df["area"].dropna().astype(str).tolist())
    if "area" in households_df.columns:
        active_das |= set(households_df["area"].dropna().astype(str).tolist())

    def _weighted_counts(df: pd.DataFrame, attr: str) -> pd.Series:
        return df[attr].value_counts().sort_index()

    def _commute_aggregate_source(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or "da_code" not in source.columns:
            return pd.DataFrame()
        aggregate_rows = source.loc[source["da_code"].astype(str).str.len() <= 4].copy()
        if aggregate_rows.empty:
            return pd.DataFrame()
        return aggregate_rows.iloc[[0]].copy()

    def _census_counts(
        table_name: str,
        attr: str,
        label_map: dict[str, str],
        collapse_map: dict[str, list[str]],
        seed_counts: pd.Series,
    ) -> tuple[pd.Series, bool, str]:
        source = context_tables.get(table_name, pd.DataFrame())
        if source.empty:
            return pd.Series(dtype=float), False, "missing"
        source_mode = "direct_da"
        if active_das and "da_code" in source.columns:
            source = source.loc[source["da_code"].astype(str).isin(active_das)].copy()
        if source.empty and table_name == "commute":
            source = _commute_aggregate_source(context_tables.get(table_name, pd.DataFrame()))
            source_mode = "aggregate_fallback" if not source.empty else "missing"
        if source.empty:
            return pd.Series(dtype=float), False, "missing"

        source_totals: dict[str, float] = {}
        matched_any = False
        for source_key, col in label_map.items():
            if col in source.columns:
                matched_any = True
                source_totals[source_key] = float(pd.to_numeric(source[col], errors="coerce").fillna(0).sum())
            else:
                source_totals[source_key] = 0.0
        if not matched_any:
            return pd.Series(dtype=float), False, "missing"

        target_to_source: dict[str, str] = {}
        source_to_targets: dict[str, list[str]] = {}
        for out_key, keys in collapse_map.items():
            source_key = keys[0] if keys else out_key
            target_to_source[out_key] = source_key
            source_to_targets.setdefault(source_key, []).append(out_key)

        values: dict[str, float] = {}
        for source_key, out_keys in source_to_targets.items():
            source_total = float(source_totals.get(source_key, 0.0))
            if len(out_keys) == 1:
                values[out_keys[0]] = source_total
                continue
            seed_subset = seed_counts.reindex(out_keys, fill_value=0.0).astype(float)
            denom = float(seed_subset.sum())
            if denom > 0:
                shares = seed_subset / denom
            else:
                shares = pd.Series(1.0 / len(out_keys), index=out_keys, dtype=float)
            for out_key in out_keys:
                values[out_key] = source_total * float(shares.loc[out_key])

        for out_key in collapse_map:
            values.setdefault(out_key, 0.0)
        if source_mode == "aggregate_fallback":
            total = float(sum(values.values()))
            target_total = float(seed_counts.sum())
            if total > 0 and target_total > 0:
                scale = target_total / total
                values = {k: v * scale for k, v in values.items()}
        return pd.Series(values, dtype=float).sort_index(), True, source_mode

    person_attr_labels = {
        "citizenship_status": PERSON_ATTR_LABELS["citizenship_status"],
        "immigrant_status": PERSON_ATTR_LABELS["immigrant_status"],
        "commute_mode": PERSON_ATTR_LABELS["commute_mode"],
        "commute_duration": PERSON_ATTR_LABELS["commute_duration"],
    }
    household_attr_labels = {
        "dwelling_type": HOUSEHOLD_ATTR_LABELS["dwelling_type"],
        "tenure": HOUSEHOLD_ATTR_LABELS["tenure"],
        "bedrooms": HOUSEHOLD_ATTR_LABELS["bedrooms"],
        "period_built": HOUSEHOLD_ATTR_LABELS["period_built"],
        "dwelling_condition": HOUSEHOLD_ATTR_LABELS["dwelling_condition"],
        "core_housing_need": HOUSEHOLD_ATTR_LABELS["core_housing_need"],
    }
    person_collapse_maps = {
        "citizenship_status": PERSON_COLLAPSE_MAP["citizenship_status"],
        "immigrant_status": PERSON_COLLAPSE_MAP["immigrant_status"],
        "commute_mode": PERSON_COLLAPSE_MAP["commute_mode"],
        "commute_duration": PERSON_COLLAPSE_MAP["commute_duration"],
    }
    household_collapse_maps = {
        "dwelling_type": HOUSEHOLD_COLLAPSE_MAP["dwelling_type"],
        "tenure": HOUSEHOLD_COLLAPSE_MAP["tenure"],
        "bedrooms": HOUSEHOLD_COLLAPSE_MAP["bedrooms"],
        "period_built": HOUSEHOLD_COLLAPSE_MAP["period_built"],
        "dwelling_condition": HOUSEHOLD_COLLAPSE_MAP["dwelling_condition"],
        "core_housing_need": HOUSEHOLD_COLLAPSE_MAP["core_housing_need"],
    }

    summary_rows = []
    detail_rows = []
    for spec in SUPPORT_SPECS:
        unit = spec.unit
        attr = spec.attribute
        df = people_df if unit == "person" else households_df
        ref_df = reference_people_df if unit == "person" else reference_households_df
        if spec.restricted_universe and attr in df.columns:
            df = df.loc[df[attr].notna()].copy()
        seed_counts = _weighted_counts(df, attr) if attr in df.columns else pd.Series(dtype=float)
        if spec.workflow_role == "core_generation":
            reference_counts = _weighted_counts(ref_df, attr) if attr in ref_df.columns else pd.Series(dtype=float)
            reference_available = not reference_counts.empty
            reference_source = "base_population"
            census_available = False
            census_source = "not_applicable"
            census_counts = pd.Series(dtype=float)
        else:
            label_map = person_attr_labels.get(attr, {}) if unit == "person" else household_attr_labels.get(attr, {})
            collapse_map = person_collapse_maps.get(attr, {}) if unit == "person" else household_collapse_maps.get(attr, {})
            reference_counts, reference_available, reference_source = _census_counts(
                spec.table_name,
                attr,
                label_map,
                collapse_map,
                seed_counts,
            )
            census_available = reference_available
            census_source = reference_source
            census_counts = reference_counts.copy()
        cats = sorted(set(seed_counts.index).union(reference_counts.index), key=lambda x: str(x))
        seed_counts = seed_counts.reindex(cats, fill_value=0.0)
        reference_counts = reference_counts.reindex(cats, fill_value=0.0)
        census_counts = census_counts.reindex(cats, fill_value=0.0)
        seed_total = float(seed_counts.sum())
        reference_total = float(reference_counts.sum()) if reference_available else np.nan
        census_total = float(census_counts.sum()) if census_available else np.nan
        seed_share = seed_counts / seed_total if seed_total > 0 else seed_counts * np.nan
        reference_share = reference_counts / reference_total if reference_available and reference_total > 0 else reference_counts * np.nan
        census_share = census_counts / census_total if census_available and census_total > 0 else census_counts * np.nan
        diff = (seed_share - reference_share) * 100.0
        abs_diff = np.abs(diff.values.astype(float))
        mae_pp = float(np.nanmean(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
        rmse_pp = float(np.sqrt(np.nanmean(diff.values.astype(float) ** 2))) if np.isfinite(diff.values.astype(float)).any() else np.nan
        max_abs_pp = float(np.nanmax(abs_diff)) if np.isfinite(abs_diff).any() else np.nan
        summary_rows.append(
            {
                "unit": unit,
                "attribute": public_attr_name(attr),
                "workflow_role": spec.workflow_role,
                "reference_available": reference_available,
                "reference_source": reference_source,
                "reference_total": reference_total,
                "census_available": census_available,
                "census_source": census_source,
                "census_total": census_total,
                "seed_total": seed_total,
                "seed_to_reference_ratio": (seed_total / reference_total) if reference_available and reference_total > 0 else np.nan,
                "seed_to_census_ratio": (seed_total / census_total) if census_available and census_total > 0 else np.nan,
                "n_categories": len(cats),
                "tvd": float(0.5 * np.nansum(np.abs((seed_share - reference_share).values))) if reference_available and reference_total > 0 else np.nan,
                "mae_pp": mae_pp,
                "rmse_pp": rmse_pp,
                "max_abs_pp": max_abs_pp,
            }
        )
        for cat in cats:
            detail_rows.append(
                {
                    "unit": unit,
                    "attribute": public_attr_name(attr),
                    "category": public_value_label(
                        attr,
                        cat,
                        age_group_scheme=age_group_scheme,
                        age_group_breaks=age_group_breaks,
                    ),
                    "workflow_role": spec.workflow_role,
                    "reference_available": reference_available,
                    "reference_source": reference_source,
                    "reference_count": float(reference_counts.loc[cat]) if reference_available else np.nan,
                    "census_available": census_available,
                    "census_source": census_source,
                    "census_count": float(census_counts.loc[cat]) if census_available else np.nan,
                    "seed_count": float(seed_counts.loc[cat]),
                    "reference_share": float(reference_share.loc[cat]) if reference_available and reference_total > 0 else np.nan,
                    "census_share": float(census_share.loc[cat]) if census_available and census_total > 0 else np.nan,
                    "seed_share": float(seed_share.loc[cat]) if seed_total > 0 else np.nan,
                    "abs_pp_diff": float(abs(diff.loc[cat])) if reference_available and reference_total > 0 and seed_total > 0 else np.nan,
                }
            )
    return pd.DataFrame(summary_rows, columns=summary_columns), pd.DataFrame(detail_rows, columns=detail_columns)


def run_full_energy_aware_workflow(
    *,
    data_root: str | Path,
    census_pumf_root: str | Path,
    housing_survey_root: str | Path | None,
    base_population_path: str | Path | None = None,
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
    method: str = "legacy_split_v1",
) -> EnergyWorkflowArtifacts:
    """Execute the end-to-end workflow for a DA selection."""
    if method == "joint_ipu_v1":
        from synthetic_population_qc.joint_fit import run_joint_ipu_workflow

        return run_joint_ipu_workflow(
            data_root=data_root,
            census_pumf_root=census_pumf_root,
            housing_survey_root=housing_survey_root,
            base_population_path=base_population_path,
            output_dir=output_dir,
            province=province,
            geography_scope=geography_scope,
            random_seed=random_seed,
            max_das=max_das,
            da_codes=da_codes,
            da_scope_name=da_scope_name,
            da_codes_file=da_codes_file,
            processed_inputs_dir=processed_inputs_dir,
            age_group_scheme=age_group_scheme,
            age_group_breaks=age_group_breaks,
            show_progress=show_progress,
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geography_scope = str(geography_scope)
    explicit_da_codes = resolve_da_scope_codes(
        da_codes=da_codes,
        da_scope_name=da_scope_name,
        da_codes_file=da_codes_file,
    )

    if processed_inputs_dir is not None:
        processed_artifacts = load_processed_artifacts(processed_inputs_dir)
        context_tables = load_processed_context_tables(processed_artifacts)
        selected_da_codes = list(explicit_da_codes) if explicit_da_codes is not None else _selected_da_codes(
            geography_scope=geography_scope,
            data_root=data_root,
            context_tables=context_tables,
            max_das=max_das,
        )
        person_seed = pd.read_parquet(processed_artifacts.person_seed_parquet)
        household_seed = pd.read_parquet(processed_artifacts.household_seed_parquet)
        artifacts = processed_artifacts
    else:
        context_tables = load_context_tables(data_root)
        selected_da_codes = list(explicit_da_codes) if explicit_da_codes is not None else _selected_da_codes(
            geography_scope=geography_scope,
            data_root=data_root,
            context_tables=context_tables,
            max_das=max_das,
        )

        processed_inputs_dir_path = out_dir / "processed_inputs"
        artifacts = export_prepared_seed_artifacts(
            data_root=data_root,
            census_pumf_root=census_pumf_root,
            housing_survey_root=housing_survey_root,
            output_dir=processed_inputs_dir_path,
            province=province,
        )
        person_seed = pd.read_parquet(artifacts.person_seed_parquet)
        household_seed = pd.read_parquet(artifacts.hierarchical_household_seed_parquet)
    person_seed = _ensure_semantic_person_schema(person_seed)
    household_seed = _ensure_semantic_household_schema(household_seed)

    support_df = build_support_assessment(
        person_seed_df=person_seed,
        household_seed_df=household_seed,
        context_tables=context_tables,
        da_codes=selected_da_codes,
    )
    support_path = out_dir / "energy_workflow_support_classification.csv"
    support_report_df = support_df.copy()
    support_report_df["attribute"] = support_report_df["attribute"].map(public_attr_name)
    if "recommended_cond_cols_json" in support_report_df.columns:
        support_report_df["recommended_cond_cols_json"] = support_report_df["recommended_cond_cols_json"].apply(
            lambda value: json.dumps(public_conditioning_cols(json.loads(value)))
        )
    if "fallback_ladder_json" in support_report_df.columns:
        support_report_df["fallback_ladder_json"] = support_report_df["fallback_ladder_json"].apply(
            lambda value: json.dumps([public_conditioning_cols(item) for item in json.loads(value)])
        )
    support_report_df.to_csv(support_path, index=False)
    support_map = support_strategy_map(support_df)
    direct_person_attrs = [
        attr
        for attr, cfg in support_map.items()
        if cfg["unit"] == "person" and cfg["assignment_route"] == "direct_person_assignment"
    ]
    sparse_person_attrs = [
        attr
        for attr, cfg in support_map.items()
        if cfg["unit"] == "person" and cfg["assignment_route"] == "sparse_fallback"
    ]
    direct_household_attrs = [
        attr
        for attr, cfg in support_map.items()
        if cfg["unit"] == "household" and cfg["assignment_route"] == "direct_household_assignment"
    ]
    sparse_household_attrs = [
        attr
        for attr, cfg in support_map.items()
        if cfg["unit"] == "household" and cfg["assignment_route"] == "sparse_fallback"
    ]

    workflow_plan = _build_workflow_plan(
        province=province,
        geography_scope=geography_scope,
        data_root=data_root,
        base_population_path=base_population_path,
        da_codes=selected_da_codes,
        support_df=support_df,
    )
    workflow_plan_path = out_dir / "energy_workflow_plan.json"
    workflow_plan_path.write_text(json.dumps(workflow_plan, indent=2), encoding="utf-8")

    base_people, base_population_metadata = _resolve_or_generate_base_population(
        data_root=data_root,
        census_pumf_root=census_pumf_root,
        output_dir=out_dir / "core_population",
        base_population_path=base_population_path,
        province=province,
        geography_scope=geography_scope,
        random_seed=random_seed,
        max_das=max_das,
        da_codes=selected_da_codes,
        show_progress=show_progress,
    )
    core_levels = _derive_core_level_maps(person_seed)
    base_people = _map_base_people_to_core_schema(base_people, core_levels)
    if selected_da_codes:
        base_people = base_people.loc[base_people["area"].astype(str).isin(selected_da_codes)].copy()
    base_people["person_uid"] = np.arange(len(base_people))
    base_households = _build_base_households(base_people)

    people_parts = []
    household_parts = []
    da_codes = [code for code in selected_da_codes if code in set(base_people["area"].dropna().astype(str).unique().tolist())]

    rng_seed = int(random_seed)

    iterator = tqdm(da_codes, desc="Proposed method DAs") if show_progress else da_codes
    sparse_rows: list[dict[str, object]] = []
    assignment_route_rows: list[dict[str, object]] = []
    for idx, da_code in enumerate(iterator):
        people_da = base_people.loc[base_people["area"] == da_code].copy()
        if people_da.empty:
            continue
        hh_da = base_households.loc[base_households["area"] == da_code].copy()
        hh_assigned = hh_da.copy()
        if not hh_da.empty:
            household_targets = _build_household_targets(hh_da, context_tables, da_code)
            household_route_rows = _assignment_route_rows_for_da(
                da_code=da_code,
                unit="household",
                attrs=direct_household_attrs + sparse_household_attrs,
                support_map=support_map,
                seed_df=household_seed,
            )
            assignment_route_rows.extend(household_route_rows)
            direct_household_attrs_da = [
                row["attribute"] for row in household_route_rows if row["selected_route"] == "direct_household_assignment"
            ]
            sparse_household_attrs_da = sorted(
                {row["attribute"] for row in household_route_rows if row["selected_route"] == "sparse_fallback"}
            )
            if direct_household_attrs_da:
                hh_dims = ["household_size", "household_type"] + direct_household_attrs_da
                hh_synth = _synthesise_records_from_seed(
                    seed_df=household_seed,
                    dims=hh_dims,
                    marginals={dim: household_targets[dim] for dim in hh_dims},
                    total_n=len(hh_da),
                )
                hh_synth["area"] = da_code
                hh_assigned = _assign_synth_values(
                    base_df=hh_da,
                    synth_df=hh_synth,
                    match_cols=["household_size", "household_type"],
                    value_cols=direct_household_attrs,
                    random_seed=rng_seed + 20000 + idx,
                )
            seed_hh = household_seed.copy()
            for attr in sparse_household_attrs_da:
                strat = support_map[attr]
                assigned, report = assign_attribute_with_fallback(
                    hh_assigned,
                    seed_df=seed_hh,
                    attr=attr,
                    target_counts=household_targets[attr],
                    row_id_col="household_id",
                    fallback_ladder=strat["fallback_ladder"],
                    min_conditional_weight=max(3.0, float(strat["min_conditional_weight"])),
                )
                hh_assigned[attr] = assigned
                sparse_rows.append(
                    {
                        "area": da_code,
                        "unit": "household",
                        "attribute": attr,
                        "support_class": strat["support_class"],
                        **report,
                    }
                )
            household_parts.append(hh_assigned)

        social_targets = _build_person_social_targets(people_da, context_tables, da_code)
        person_route_rows = _assignment_route_rows_for_da(
            da_code=da_code,
            unit="person",
            attrs=direct_person_attrs + sparse_person_attrs,
            support_map=support_map,
            seed_df=person_seed,
        )
        assignment_route_rows.extend(person_route_rows)
        direct_person_attrs_da = [
            row["attribute"] for row in person_route_rows if row["selected_route"] == "direct_person_assignment"
        ]
        sparse_person_attrs_da = sorted(
            {row["attribute"] for row in person_route_rows if row["selected_route"] == "sparse_fallback"}
        )

        people_assigned = people_da.copy()
        if direct_person_attrs_da:
            social_dims = ["sex", "age_group", "labour_force_status"] + direct_person_attrs_da
            social_synth = _synthesise_records_from_seed(
                seed_df=person_seed,
                dims=social_dims,
                marginals={dim: social_targets[dim] for dim in social_dims},
                total_n=len(people_da),
            )
            social_synth["area"] = da_code
            people_assigned = _assign_synth_values(
                base_df=people_da,
                synth_df=social_synth,
                match_cols=["sex", "age_group", "labour_force_status"],
                value_cols=direct_person_attrs_da,
                random_seed=rng_seed + idx,
            )
        for attr in [x for x in sparse_person_attrs_da if x in {"citizenship_status", "immigrant_status"}]:
            strat = support_map[attr]
            assigned, report = assign_attribute_with_fallback(
                people_assigned,
                seed_df=person_seed,
                attr=attr,
                target_counts=social_targets[attr],
                row_id_col="person_uid",
                fallback_ladder=strat["fallback_ladder"],
                min_conditional_weight=max(3.0, float(strat["min_conditional_weight"])),
            )
            people_assigned[attr] = assigned
            sparse_rows.append(
                {
                    "area": da_code,
                    "unit": "person",
                    "attribute": attr,
                    "support_class": strat["support_class"],
                    **report,
                }
            )

        employed_da, mobility_targets = _build_person_mobility_targets(people_da, context_tables, da_code)
        if mobility_targets:
            employed_assigned = people_assigned.loc[people_assigned["labour_force_status"] == 1867].copy()
            mobility_seed = _with_alias_columns(
                person_seed.loc[
                    person_seed["labour_force_status"] == 1867,
                    ["weight", "sex", "age_group", "commute_mode_group", "commute_duration"],
                ],
                {"commute_mode": "commute_mode_group"},
            )
            for attr in [x for x in ["commute_mode", "commute_duration"] if x in sparse_person_attrs_da]:
                strat = support_map.get(attr, {})
                assigned, report = assign_attribute_with_fallback(
                    employed_assigned,
                    seed_df=mobility_seed,
                    attr=attr,
                    target_counts=mobility_targets[attr],
                    row_id_col="person_uid",
                    fallback_ladder=strat.get("fallback_ladder", [["sex", "age_group"], ["sex"], []]),
                    min_conditional_weight=max(3.0, float(strat.get("min_conditional_weight", 10.0))),
                )
                employed_assigned[attr] = assigned
                sparse_rows.append(
                    {
                        "area": da_code,
                        "unit": "person",
                        "attribute": attr,
                        "support_class": strat.get("support_class", "moderately_sparse"),
                        **report,
                    }
                )
            people_assigned = people_assigned.merge(
                employed_assigned[["person_uid", "commute_mode", "commute_duration"]],
                on="person_uid",
                how="left",
            )
        else:
            people_assigned["commute_mode"] = pd.NA
            people_assigned["commute_duration"] = pd.NA
        people_parts.append(people_assigned)

    people_out = pd.concat(people_parts, ignore_index=True)
    households_out = pd.concat(household_parts, ignore_index=True)

    people_path = out_dir / "energy_workflow_people.parquet"
    households_path = out_dir / "energy_workflow_households.parquet"

    household_coherence_df = _build_household_coherence_audit(households_out)
    people_public = _canonicalize_people_output(
        people_out,
        age_group_scheme=age_group_scheme,
        age_group_breaks=age_group_breaks,
    )
    households_public = _canonicalize_households_output(households_out)
    people_public.to_parquet(people_path, index=False)
    households_public.to_parquet(households_path, index=False)

    household_coherence_path = out_dir / "energy_workflow_household_coherence_audit.csv"
    household_coherence_df.to_csv(household_coherence_path, index=False)

    summary_df, details_df = _summarize_aligned_outputs(
        people_out,
        households_out,
        context_tables,
        reference_people_df=base_people,
        reference_households_df=base_households,
        age_group_scheme=age_group_scheme,
        age_group_breaks=age_group_breaks,
    )
    summary_path = out_dir / "energy_workflow_metric_summary.csv"
    details_path = out_dir / "energy_workflow_metric_details.csv"
    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    results_summary_df = build_results_summary(summary_df)
    results_summary_path = out_dir / "energy_workflow_results_summary.csv"
    results_summary_df.to_csv(results_summary_path, index=False)

    sparse_report_columns = [
        "area",
        "unit",
        "attribute",
        "support_class",
        "fallback_rank",
        "chosen_conditioning_json",
        "used_global_fallback",
    ]
    sparse_report_df = pd.DataFrame(sparse_rows, columns=sparse_report_columns)
    if not sparse_report_df.empty:
        sparse_report_df["attribute"] = sparse_report_df["attribute"].map(public_attr_name)
        if "chosen_conditioning_json" in sparse_report_df.columns:
            sparse_report_df["chosen_conditioning_json"] = sparse_report_df["chosen_conditioning_json"].apply(
                lambda value: json.dumps(public_conditioning_cols(json.loads(value)))
            )
    sparse_report_path = out_dir / "energy_workflow_sparse_handling.csv"
    sparse_report_df.to_csv(sparse_report_path, index=False)

    assignment_route_columns = [
        "area",
        "unit",
        "attribute",
        "planned_route",
        "selected_route",
        "recommended_conditioning_json",
        "planned_min_conditional_weight",
        "observed_min_conditional_weight",
        "downgraded_to_sparse",
        "public_attribute",
    ]
    assignment_route_df = pd.DataFrame(assignment_route_rows, columns=assignment_route_columns)
    if not assignment_route_df.empty and "public_attribute" in assignment_route_df.columns:
        assignment_route_df["attribute"] = assignment_route_df["public_attribute"]
        assignment_route_df = assignment_route_df.drop(columns=["public_attribute"])
    assignment_route_path = out_dir / "energy_workflow_assignment_route_decisions.csv"
    assignment_route_df.to_csv(assignment_route_path, index=False)

    metadata = {
        "province": province,
        "geography_scope": geography_scope,
        "da_scope_name": da_scope_name,
        "da_codes_file": str(Path(da_codes_file)) if da_codes_file is not None else None,
        "data_root": str(Path(data_root)),
        "base_population_path": str(Path(base_population_path)) if base_population_path is not None else None,
        "processed_inputs_dir": str(Path(processed_inputs_dir)) if processed_inputs_dir is not None else None,
        "base_population_mode": base_population_metadata.get("mode"),
        "base_population_metadata": base_population_metadata,
        "n_people": int(len(people_out)),
        "n_households": int(len(households_out)),
        "n_das": int(len(da_codes)),
        "selected_da_count": int(len(selected_da_codes)),
        "selected_da_codes": selected_da_codes,
        "age_group_scheme": age_group_scheme,
        "age_group_breaks": list(age_group_breaks) if age_group_breaks is not None else None,
        "core_generation_attributes": sorted(
            [
                public_attr_name(attr)
                for attr, cfg in support_map.items()
                if cfg.get("workflow_role") == "core_generation"
            ]
        ),
        "direct_person_attributes": sorted([public_attr_name(attr) for attr in direct_person_attrs]),
        "direct_household_attributes": sorted([public_attr_name(attr) for attr in direct_household_attrs]),
        "sparse_person_attributes": sorted([public_attr_name(attr) for attr in sparse_person_attrs]),
        "sparse_household_attributes": sorted([public_attr_name(attr) for attr in sparse_household_attrs]),
        "all_workflow_attributes": sorted([public_attr_name(attr) for attr in support_map.keys()]),
        "evaluated_attributes": sorted(summary_df["attribute"].astype(str).unique().tolist()) if not summary_df.empty else [],
        "workflow_steps": workflow_plan["workflow_steps"],
        "workflow_plan_json": str(workflow_plan_path),
        "assignment_route_csv": str(assignment_route_path),
        "household_coherence_csv": str(household_coherence_path),
        "n_household_coherence_issues": int(len(household_coherence_df)),
        "processed_input_artifacts": {k: str(v) for k, v in asdict(artifacts).items()},
    }
    metadata_path = out_dir / "energy_workflow_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return EnergyWorkflowArtifacts(
        output_dir=out_dir,
        people_parquet=people_path,
        households_parquet=households_path,
        workflow_plan_json=workflow_plan_path,
        assignment_route_csv=assignment_route_path,
        household_coherence_csv=household_coherence_path,
        summary_csv=summary_path,
        details_csv=details_path,
        results_summary_csv=results_summary_path,
        support_classification_csv=support_path,
        sparse_handling_csv=sparse_report_path,
        metadata_json=metadata_path,
    )

def main() -> int:
    """Run the workflow from the command line."""
    parser = argparse.ArgumentParser(description="Run the end-to-end energy-aware synthetic workflow.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--census-pumf-root", required=True)
    parser.add_argument("--housing-survey-root", default=None)
    parser.add_argument("--base-population-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--province", default="24")
    parser.add_argument("--geography-scope", default="montreal")
    parser.add_argument("--method", default="joint_ipu_v1")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-das", type=int, default=None)
    args = parser.parse_args()

    artifacts = run_full_energy_aware_workflow(
        data_root=args.data_root,
        census_pumf_root=args.census_pumf_root,
        housing_survey_root=args.housing_survey_root,
        base_population_path=args.base_population_path,
        output_dir=args.output_dir,
        province=args.province,
        geography_scope=args.geography_scope,
        method=args.method,
        random_seed=args.random_seed,
        max_das=args.max_das,
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(artifacts).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
