"""Prepare harmonized person and household seeds from Canadian microdata inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_population_qc.workflow_inputs import (
    build_workflow_input_contract,
    summarize_workflow_input_contract,
)
from synthetic_population_qc.seed_transforms import (
    map_age_grp,
    map_cfstat,
    map_hdgree,
    map_hhsize,
    map_lfact,
    map_prihm,
    map_totinc,
)


PERSON_USECOLS = [
    "PPSORT",
    "AGEGRP",
    "Gender",
    "HDGREE",
    "LFACT",
    "TotInc",
    "HHSIZE",
    "CFSTAT",
    "PRIHM",
    "Citizen",
    "IMMSTAT",
    "MODE",
    "PWDUR",
    "DTYPE",
    "Tenur",
    "BedRm",
    "REPAIR",
    "HCORENEED_IND",
    "PR",
    "WEIGHT",
]

HIER_USECOLS = [
    "HH_ID",
    "PP_ID",
    "AGEGRP",
    "GENDER",
    "HDGREE",
    "LFACT",
    "TOTINC",
    "CFSTAT",
    "PRIHM",
    "HHMAINP",
    "CITIZEN",
    "IMMSTAT",
    "MODE",
    "PWDUR",
    "DTYPE",
    "TENUR",
    "BEDRM",
    "BUILT",
    "REPAIR",
    "HCORENEED_IND",
    "PR",
    "WEIGHT",
]

CHS_USECOLS = [
    "PUMFID",
    "PHHSIZE",
    "PHHTTINC",
    "PDWLTYPE",
    "PDV_SUIT",
    "P1DCT_20",
    "PRSPIMST",
    "PRSPGNDR",
    "POWN_20",
    "PFWEIGHT",
]


@dataclass(frozen=True)
class PreparedSeedArtifacts:
    """Seed-preparation outputs written to one processed directory."""
    output_dir: Path
    input_contract_csv: Path
    person_seed_parquet: Path
    hierarchical_household_seed_parquet: Path
    chs_household_seed_parquet: Path | None
    person_summary_csv: Path
    household_summary_csv: Path
    metadata_json: Path


def _coerce_province(series: pd.Series) -> pd.Series:
    """Normalize province codes to stripped string form."""
    return series.astype(str).str.strip()


def _map_citizenship_status(series: pd.Series) -> pd.Series:
    """Collapse raw citizenship codes to the binary workflow category."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.isin([1, 2]), x.eq(3)],
            ["canadian_citizen", "not_canadian_citizen"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_citizenship_detail(series: pd.Series) -> pd.Series:
    """Map raw citizenship codes to a more detailed public-facing category."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3)],
            ["citizen_by_birth", "citizen_by_naturalization", "not_canadian_citizen"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_immigrant_status(series: pd.Series) -> pd.Series:
    """Collapse raw immigration codes to the harmonized workflow categories."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3)],
            ["non_immigrant", "immigrant", "non_permanent_resident"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_commute_mode(series: pd.Series) -> pd.Series:
    """Map detailed commute-mode codes to harmonized seed categories."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                x.eq(1),
                x.eq(2),
                x.eq(5),
                x.eq(6),
                x.eq(7),
                x.eq(3),
                x.eq(4),
            ],
            [
                "bicycle",
                "car_driver",
                "car_passenger",
                "public_transit",
                "walked",
                "motorcycle_scooter_moped",
                "other_method",
            ],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_commute_mode_group(series: pd.Series) -> pd.Series:
    """Collapse detailed commute modes into the public comparison groups."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.isin([2, 5, 3]), x.eq(6), x.isin([1, 7]), x.eq(4)],
            ["private_vehicle", "public_transit", "active_transport", "other_method"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_commute_duration(series: pd.Series) -> pd.Series:
    """Map commute-duration codes to harmonized duration buckets."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3), x.eq(4), x.eq(5)],
            [
                "lt_15_min",
                "15_to_29_min",
                "30_to_44_min",
                "45_to_59_min",
                "60_plus_min",
            ],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_dwelling_type_census(series: pd.Series) -> pd.Series:
    """Collapse dwelling-type codes to the workflow household categories."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3)],
            ["single_detached_house", "apartment", "other_dwelling"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_tenure_census(series: pd.Series) -> pd.Series:
    """Collapse tenure codes to the workflow household categories."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([x.eq(1), x.eq(2)], ["owner", "renter_or_band"], default=None),
        index=series.index,
        dtype="object",
    )


def _map_bedrooms_census(series: pd.Series) -> pd.Series:
    """Collapse bedroom counts to the workflow bedroom buckets."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(0), x.eq(1), x.eq(2), x.eq(3), x.ge(4)],
            ["no_bedroom", "1_bedroom", "2_bedrooms", "3_bedrooms", "4plus_bedrooms"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_repair_census(series: pd.Series) -> pd.Series:
    """Collapse repair codes to the workflow dwelling-condition buckets."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3)],
            ["regular_maintenance", "minor_repairs", "major_repairs"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _map_core_need_census(series: pd.Series) -> pd.Series:
    """Map core-housing-need indicators to the workflow labels."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([x.eq(0), x.eq(100)], ["not_in_core_need", "in_core_need"], default=None),
        index=series.index,
        dtype="object",
    )


def _map_built_census(series: pd.Series) -> pd.Series:
    """Map period-built codes to the workflow construction-period labels."""
    x = pd.to_numeric(series, errors="coerce")
    labels = {
        1: "1945_or_before",
        2: "1946_to_1960",
        3: "1961_to_1970",
        4: "1971_to_1980",
        5: "1981_to_1990",
        6: "1991_to_1995",
        7: "1996_to_2000",
        8: "2001_to_2005",
        9: "2006_to_2010",
        10: "2011_to_2015",
        11: "2016_to_2021",
    }
    return x.map(labels).astype("object")


def _map_chs_gender(series: pd.Series) -> pd.Series:
    """Map CHS respondent gender codes to harmonized labels."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([x.eq(1), x.eq(2), x.eq(3)], ["male", "female", "other_or_specified"], default=None),
        index=series.index,
        dtype="object",
    )


def _map_chs_immigration(series: pd.Series) -> pd.Series:
    """Map CHS respondent immigration codes to harmonized labels."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([x.eq(1), x.eq(2)], ["non_immigrant", "immigrant_or_npr"], default=None),
        index=series.index,
        dtype="object",
    )


def _map_chs_suitability(series: pd.Series) -> pd.Series:
    """Map CHS suitability codes to workflow housing-suitability labels."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select([x.eq(1), x.eq(2)], ["suitable", "not_suitable"], default=None),
        index=series.index,
        dtype="object",
    )


def _map_chs_bedrooms(series: pd.Series) -> pd.Series:
    """Map CHS bedroom categories to compact bedroom labels."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [x.eq(1), x.eq(2), x.eq(3), x.eq(4)],
            ["one_or_fewer", "two", "three", "four_or_more"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _bucket_household_income_raw(series: pd.Series) -> pd.Series:
    """Bucket raw household income into broad CHS-side enrichment bands."""
    x = pd.to_numeric(series, errors="coerce")
    return pd.Series(
        np.select(
            [
                x.lt(30000),
                x.ge(30000) & x.lt(80000),
                x.ge(80000) & x.lt(150000),
                x.ge(150000),
            ],
            ["lt_30k", "30k_to_79k", "80k_to_149k", "150k_plus"],
            default=None,
        ),
        index=series.index,
        dtype="object",
    )


def _prepare_core_seed(df: pd.DataFrame, *, gender_col: str, totinc_col: str) -> pd.DataFrame:
    """Rename raw PUMF columns and apply the core demographic harmonization maps."""
    work = df.copy()
    work = work.rename(
        columns={
            "AGEGRP": "agegrp",
            "HDGREE": "hdgree",
            "LFACT": "lfact",
            "HHSIZE": "hhsize",
            "CFSTAT": "cfstat",
            "PRIHM": "prihm",
            gender_col: "Gender",
            totinc_col: "TotInc",
        }
    )
    work = map_age_grp(work)
    work = map_hdgree(work)
    work = map_lfact(work)
    work = map_hhsize(work)
    work = map_totinc(work)
    work = map_cfstat(work)
    work = map_prihm(work)
    return work


def load_census_individual_pumf(csv_path: str | Path, *, province: str = "24") -> pd.DataFrame:
    """Load and province-filter the person-level Census PUMF."""
    df = pd.read_csv(csv_path, usecols=PERSON_USECOLS)
    df = df.loc[_coerce_province(df["PR"]) == str(province)].copy()
    return df


def load_census_hierarchical_pumf(csv_path: str | Path, *, province: str = "24") -> pd.DataFrame:
    """Load and province-filter the hierarchical household Census PUMF."""
    df = pd.read_csv(csv_path, usecols=HIER_USECOLS)
    df = df.loc[_coerce_province(df["PR"]) == str(province)].copy()
    return df


def load_chs_pumf(csv_path: str | Path) -> pd.DataFrame:
    """Load the subset of CHS columns used by the workflow."""
    return pd.read_csv(csv_path, usecols=CHS_USECOLS)


def prepare_person_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize person-level PUMF records into the workflow seed schema."""
    core = _prepare_core_seed(df, gender_col="Gender", totinc_col="TotInc")
    out = pd.DataFrame(
        {
            "person_id": core["PPSORT"].astype("Int64"),
            "weight": pd.to_numeric(core["WEIGHT"], errors="coerce"),
            "sex": core["Gender"].map({1: "male", 2: "female"}),
            "prihm": pd.to_numeric(core["prihm"], errors="coerce"),
            # Keep semantic names as the active workflow schema.
            "age_group": pd.to_numeric(core["agegrp"], errors="coerce"),
            "education_level": pd.to_numeric(core["hdgree"], errors="coerce"),
            "labour_force_status": pd.to_numeric(core["lfact"], errors="coerce"),
            "household_size": pd.to_numeric(core["hhsize"], errors="coerce").clip(lower=1, upper=5),
            "household_income": pd.to_numeric(core["TotInc"], errors="coerce"),
            "family_status": pd.to_numeric(core["cfstat"], errors="coerce"),
            # Retain encoded aliases only while downstream migrations finish.
            "agegrp_core": pd.to_numeric(core["agegrp"], errors="coerce"),
            "hdgree_core": pd.to_numeric(core["hdgree"], errors="coerce"),
            "lfact_core": pd.to_numeric(core["lfact"], errors="coerce"),
            "hhsize_core": pd.to_numeric(core["hhsize"], errors="coerce"),
            "totinc_core": pd.to_numeric(core["TotInc"], errors="coerce"),
            "cfstat_core": pd.to_numeric(core["cfstat"], errors="coerce"),
            "citizenship_status": _map_citizenship_status(core["Citizen"]),
            "citizenship_detail": _map_citizenship_detail(core["Citizen"]),
            "immigrant_status": _map_immigrant_status(core["IMMSTAT"]),
            "commute_mode": _map_commute_mode(core["MODE"]),
            "commute_mode_group": _map_commute_mode_group(core["MODE"]),
            "commute_duration": _map_commute_duration(core["PWDUR"]),
            "dwelling_type_proxy": _map_dwelling_type_census(core["DTYPE"]),
            "tenure_proxy": _map_tenure_census(core["Tenur"]),
            "bedrooms_proxy": _map_bedrooms_census(core["BedRm"]),
            "dwelling_condition_proxy": _map_repair_census(core["REPAIR"]),
            "core_housing_need_proxy": _map_core_need_census(core["HCORENEED_IND"]),
            "source_dataset": "2021_census_individual_pumf",
        }
    )
    return out.reset_index(drop=True)


def _select_household_reference_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one reference member per household for household-side seed fields."""
    work = df.copy()
    work["_priority"] = 2
    work.loc[pd.to_numeric(work["PRIHM"], errors="coerce").eq(1), "_priority"] = 0
    work.loc[pd.to_numeric(work["HHMAINP"], errors="coerce").eq(1), "_priority"] = 1
    work = work.sort_values(["HH_ID", "_priority", "PP_ID"]).drop_duplicates("HH_ID", keep="first")
    return work.drop(columns="_priority")


def _derive_household_type_from_members(df: pd.DataFrame) -> pd.DataFrame:
    """Infer harmonized household types from member CFSTAT compositions."""
    work = df.copy()
    grouped = work.groupby("HH_ID", sort=False)
    rows = []
    for hh_id, g in grouped:
        cfstats = pd.to_numeric(g["CFSTAT"], errors="coerce").dropna().astype(int).tolist()
        vals = set(cfstats)
        n = int(len(g))
        if n == 1:
            hh_type = "one_person"
        elif n == 2 and vals.issubset({1}):
            # Align with the DA census target: couple family without children
            # and without additional persons.
            hh_type = "couple_without_children"
        elif (
            n >= 3
            and set(cfstats).issubset({2, 4})
            and cfstats.count(2) == 2
            and cfstats.count(4) >= 1
        ):
            # Couple family with children, without additional persons.
            hh_type = "couple_with_children"
        elif (
            n >= 2
            and set(cfstats).issubset({3, 5})
            and cfstats.count(3) == 1
            and cfstats.count(5) >= 1
        ):
            # One-parent family, without additional persons.
            hh_type = "one_parent"
        else:
            hh_type = "other"
        rows.append({"HH_ID": hh_id, "household_type": hh_type, "n_members": n})
    return pd.DataFrame(rows)


def prepare_hierarchical_household_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize hierarchical PUMF records into the workflow household seed."""
    reference = _select_household_reference_rows(df)
    household_types = _derive_household_type_from_members(df)
    out = reference.merge(household_types, on="HH_ID", how="left")
    out["household_size"] = (
        pd.to_numeric(out["n_members"], errors="coerce")
        .clip(lower=1)
        .fillna(1)
        .astype(int)
        .clip(upper=5)
        .map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"})
    )
    seed = pd.DataFrame(
        {
            "household_id": out["HH_ID"].astype(str),
            "reference_person_id": out["PP_ID"].astype(str),
            "weight": pd.to_numeric(out["WEIGHT"], errors="coerce"),
            "household_type": out["household_type"],
            "household_size": out["household_size"],
            "dwelling_type": _map_dwelling_type_census(out["DTYPE"]),
            "tenure": _map_tenure_census(out["TENUR"]),
            "bedrooms": _map_bedrooms_census(out["BEDRM"]),
            "period_built": _map_built_census(out["BUILT"]),
            "dwelling_condition": _map_repair_census(out["REPAIR"]),
            "core_housing_need": _map_core_need_census(out["HCORENEED_IND"]),
            "citizenship_status_ref": _map_citizenship_status(out["CITIZEN"]),
            "immigrant_status_ref": _map_immigrant_status(out["IMMSTAT"]),
            "commute_mode_ref": _map_commute_mode(out["MODE"]),
            "commute_duration_ref": _map_commute_duration(out["PWDUR"]),
            "source_dataset": "2021_census_hierarchical_pumf",
        }
    )
    return seed.reset_index(drop=True)


def prepare_chs_household_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize CHS records into comparable household-side enrichment fields."""
    out = pd.DataFrame(
        {
            "chs_id": df["PUMFID"].astype(str),
            "weight": pd.to_numeric(df["PFWEIGHT"], errors="coerce"),
            "household_size": (
                pd.to_numeric(df["PHHSIZE"], errors="coerce")
                .clip(lower=1)
                .fillna(np.nan)
                .clip(upper=5)
                .map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5plus"})
            ),
            "household_income_band": _bucket_household_income_raw(df["PHHTTINC"]),
            "dwelling_type_chs_code": df["PDWLTYPE"].astype(str),
            "housing_suitability": _map_chs_suitability(df["PDV_SUIT"]),
            "bedrooms": _map_chs_bedrooms(df["P1DCT_20"]),
            "mortgage_present": pd.to_numeric(df["POWN_20"], errors="coerce").map({1: "yes", 2: "no"}),
            "respondent_gender": _map_chs_gender(df["PRSPGNDR"]),
            "respondent_immigration_status": _map_chs_immigration(df["PRSPIMST"]),
            "source_dataset": "2022_canadian_housing_survey_pumf",
        }
    )
    return out.reset_index(drop=True)


def _summarize_seed(df: pd.DataFrame, *, seed_name: str) -> pd.DataFrame:
    """Summarize coverage and top-category diagnostics for one prepared seed."""
    rows = []
    for col in df.columns:
        if col in {"person_id", "household_id", "reference_person_id", "chs_id", "source_dataset"}:
            continue
        s = df[col].dropna()
        rows.append(
            {
                "seed_name": seed_name,
                "column": col,
                "n_non_null": int(s.shape[0]),
                "coverage": float(s.shape[0] / len(df)) if len(df) else np.nan,
                "n_unique": int(s.nunique()) if not s.empty else 0,
                "top_value": s.astype(str).value_counts().index[0] if not s.empty else None,
                "top_share": float(s.astype(str).value_counts(normalize=True).iloc[0]) if not s.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def export_prepared_seed_artifacts(
    *,
    data_root: str | Path,
    census_pumf_root: str | Path,
    housing_survey_root: str | Path | None = None,
    output_dir: str | Path,
    province: str = "24",
) -> PreparedSeedArtifacts:
    """Prepare and persist the harmonized person/household seed artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = build_workflow_input_contract(
        data_root,
        census_pumf_root=census_pumf_root,
        housing_survey_root=housing_survey_root,
    )
    contract_df = summarize_workflow_input_contract(
        data_root,
        census_pumf_root=census_pumf_root,
        housing_survey_root=housing_survey_root,
    )
    contract_csv = out_dir / "input_contract.csv"
    contract_df.to_csv(contract_csv, index=False)

    if not contract.individual_pumf_candidates:
        raise FileNotFoundError("No individual census PUMF candidate found.")
    if not contract.household_pumf_candidates:
        raise FileNotFoundError("No hierarchical/household census PUMF candidate found.")

    individual_df = load_census_individual_pumf(contract.individual_pumf_candidates[0], province=province)
    hierarchical_df = load_census_hierarchical_pumf(contract.household_pumf_candidates[0], province=province)
    person_seed = prepare_person_seed(individual_df)
    hierarchical_seed = prepare_hierarchical_household_seed(hierarchical_df)

    person_parquet = out_dir / "person_seed.parquet"
    hierarchical_parquet = out_dir / "household_seed_hierarchical.parquet"
    person_seed.to_parquet(person_parquet, index=False)
    hierarchical_seed.to_parquet(hierarchical_parquet, index=False)

    chs_parquet: Path | None = None
    chs_summary = pd.DataFrame()
    chs_rows = 0
    if contract.housing_survey_candidates:
        chs_df = load_chs_pumf(contract.housing_survey_candidates[0])
        chs_seed = prepare_chs_household_seed(chs_df)
        chs_rows = int(len(chs_seed))
        chs_parquet = out_dir / "household_seed_chs.parquet"
        chs_seed.to_parquet(chs_parquet, index=False)
        chs_summary = _summarize_seed(chs_seed, seed_name="chs_household")

    person_summary = _summarize_seed(person_seed, seed_name="person")
    household_summary = pd.concat(
        [
            _summarize_seed(hierarchical_seed, seed_name="hierarchical_household"),
            chs_summary,
        ],
        ignore_index=True,
    )
    person_summary_csv = out_dir / "person_seed_summary.csv"
    household_summary_csv = out_dir / "household_seed_summary.csv"
    person_summary.to_csv(person_summary_csv, index=False)
    household_summary.to_csv(household_summary_csv, index=False)

    metadata = {
        "province": province,
        "data_root": str(Path(data_root)),
        "census_pumf_root": str(Path(census_pumf_root)),
        "housing_survey_root": str(Path(housing_survey_root)) if housing_survey_root is not None else None,
        "person_rows": int(len(person_seed)),
        "hierarchical_household_rows": int(len(hierarchical_seed)),
        "chs_household_rows": chs_rows,
        "input_contract": contract_df.to_dict(orient="records"),
    }
    metadata_json = out_dir / "prepared_seed_metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return PreparedSeedArtifacts(
        output_dir=out_dir,
        input_contract_csv=contract_csv,
        person_seed_parquet=person_parquet,
        hierarchical_household_seed_parquet=hierarchical_parquet,
        chs_household_seed_parquet=chs_parquet,
        person_summary_csv=person_summary_csv,
        household_summary_csv=household_summary_csv,
        metadata_json=metadata_json,
    )


def _default_output_dir() -> Path:
    """Return the default output directory for prepared seed artifacts."""
    return Path("data") / "processed" / "synthetic_population" / "prepared_inputs"


def main() -> int:
    """Run seed preparation from the command line."""
    parser = argparse.ArgumentParser(description="Prepare harmonized person and household seed artifacts for the workflow.")
    parser.add_argument("--data-root", required=True, help="Path to urban-energy-data repository root.")
    parser.add_argument("--census-pumf-root", required=True, help="Path containing the 2021 Census ind/heir PUMFs.")
    parser.add_argument("--housing-survey-root", default=None, help="Optional path to the 2022 CHS PUMF folder.")
    parser.add_argument("--output-dir", default=str(_default_output_dir()), help="Output directory for prepared seed artifacts.")
    parser.add_argument("--province", default="24", help="Province code to filter on. Defaults to Quebec (24).")
    args = parser.parse_args()

    artifacts = export_prepared_seed_artifacts(
        data_root=args.data_root,
        census_pumf_root=args.census_pumf_root,
        housing_survey_root=args.housing_survey_root,
        output_dir=args.output_dir,
        province=args.province,
    )
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in asdict(artifacts).items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
