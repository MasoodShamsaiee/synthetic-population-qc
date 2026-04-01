from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_num(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        if x in ("x", "F", ".."):
            return None
        return float(x)
    except Exception:
        return None


def _safe_div(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return np.nan
    return float(num) / float(den)


def _sum_varids(census_da: pd.DataFrame, var_ids: list[int]) -> float:
    vals = []
    for vid in var_ids:
        s = census_da.loc[census_da["variableId"] == vid, "total"]
        if s.empty:
            continue
        v = _to_num(s.iloc[0])
        if v is not None:
            vals.append(v)
    return float(np.sum(vals)) if vals else np.nan


def _census_income_bins(census_da: pd.DataFrame, totinc_vb: dict[int, int]) -> dict[str, float]:
    # same grouping logic used in original notebook code
    out = {
        "inc_lt_20k": np.nan,
        "inc_20k_60k": np.nan,
        "inc_60k_100k": np.nan,
        "inc_ge_100k": np.nan,
        "inc_15plus_total": np.nan,
    }
    if 0 not in totinc_vb or 1 not in totinc_vb or 2 not in totinc_vb or 3 not in totinc_vb:
        return out

    v0 = _sum_varids(census_da, [totinc_vb[0], totinc_vb[0] + 1])
    v1 = _sum_varids(census_da, [totinc_vb[1] + i for i in range(4)])
    v2 = _sum_varids(census_da, [totinc_vb[2] + i for i in range(4)])
    v3 = _sum_varids(census_da, [totinc_vb[3]])
    out["inc_lt_20k"] = v0
    out["inc_20k_60k"] = v1
    out["inc_60k_100k"] = v2
    out["inc_ge_100k"] = v3
    # optional explicit total if key exists
    if 4 in totinc_vb:
        out["inc_15plus_total"] = _sum_varids(census_da, [totinc_vb[4]])
    return out


def _syn_income_bins(df_da: pd.DataFrame) -> dict[str, float]:
    out = {
        "inc_lt_20k": np.nan,
        "inc_20k_60k": np.nan,
        "inc_60k_100k": np.nan,
        "inc_ge_100k": np.nan,
        "inc_15plus_total": np.nan,
    }
    if "age" not in df_da.columns or "totinc" not in df_da.columns:
        return out
    age = pd.to_numeric(df_da["age"], errors="coerce")
    inc = pd.to_numeric(df_da["totinc"], errors="coerce")
    m15 = age >= 15
    out["inc_15plus_total"] = float(m15.sum())
    out["inc_lt_20k"] = float(((inc == 0) & m15).sum())
    out["inc_20k_60k"] = float(((inc == 1) & m15).sum())
    out["inc_60k_100k"] = float(((inc == 2) & m15).sum())
    out["inc_ge_100k"] = float(((inc == 3) & m15).sum())
    return out


def _syn_household_type_counts(df_da: pd.DataFrame) -> dict[str, float]:
    out = {
        "hh_couple_wo_child": np.nan,
        "hh_couple_w_child": np.nan,
        "hh_one_parent": np.nan,
        "hh_one_person": np.nan,
        "hh_other": np.nan,
    }
    if "HID" not in df_da.columns or "hhtype" not in df_da.columns:
        return out
    hh = df_da.loc[df_da["HID"] != -1].drop_duplicates("HID")
    ht = pd.to_numeric(hh["hhtype"], errors="coerce")
    out["hh_couple_wo_child"] = float((ht == 0).sum())
    out["hh_couple_w_child"] = float((ht == 1).sum())
    out["hh_one_parent"] = float((ht == 2).sum())
    out["hh_one_person"] = float((ht == 3).sum())
    out["hh_other"] = float((ht == 4).sum())
    return out


def _census_household_type_counts(census_da: pd.DataFrame, cfstat_vb: dict[int, int]) -> dict[str, float]:
    # Mapping based on your current notebook_functions.load_vbs_ids
    out = {
        "hh_couple_wo_child": np.nan,
        "hh_couple_w_child": np.nan,
        "hh_one_parent": np.nan,
        "hh_one_person": np.nan,
        "hh_other": np.nan,
    }
    if not all(k in cfstat_vb for k in [0, 1, 2, 3, 4]):
        return out
    v0 = _sum_varids(census_da, [cfstat_vb[0]])
    v1 = _sum_varids(census_da, [cfstat_vb[1]])
    v2 = _sum_varids(census_da, [cfstat_vb[2]])
    v3 = _sum_varids(census_da, [cfstat_vb[3]])  # non-census-family in your 2021 mapping
    v4 = _sum_varids(census_da, [cfstat_vb[4]])  # one-person in your 2021 mapping
    out["hh_couple_wo_child"] = v0
    out["hh_couple_w_child"] = v1
    out["hh_one_parent"] = v2
    out["hh_one_person"] = v4
    out["hh_other"] = v3
    return out


def build_quality_summary_table(
    syn_with_hh: pd.DataFrame,
    census: pd.DataFrame,
    *,
    total_vb_id: int,
    total_age_by_sex_vb_id: int,
    total_hh_vb_id: int,
    hhsize_vb: dict[int, int],
    totinc_vb: dict[int, int],
    cfstat_vb: dict[int, int],
    area_col: str = "area",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Build DA-level quality table comparing synthetic population/households to census targets.
    """
    from tqdm.auto import tqdm

    work = syn_with_hh.copy()
    work[area_col] = work[area_col].astype(str)
    das = sorted(work[area_col].dropna().unique().tolist())

    rows: list[dict] = []
    iterator = tqdm(das, desc="Quality summary by DA") if show_progress else das
    for da in iterator:
        df_da = work.loc[work[area_col] == da].copy()
        census_da = census.loc[census["geocode"].astype(str) == str(da)].copy()
        if census_da.empty or df_da.empty:
            continue

        pop_c = _sum_varids(census_da, [total_vb_id])
        pop_age_c = _sum_varids(census_da, [total_age_by_sex_vb_id])
        hh_c = _sum_varids(census_da, [total_hh_vb_id])
        male_c = _to_num(census_da.loc[census_da["variableId"] == total_age_by_sex_vb_id, "totalMale"].iloc[0]) if not census_da.loc[census_da["variableId"] == total_age_by_sex_vb_id].empty else np.nan
        female_c = _to_num(census_da.loc[census_da["variableId"] == total_age_by_sex_vb_id, "totalFemale"].iloc[0]) if not census_da.loc[census_da["variableId"] == total_age_by_sex_vb_id].empty else np.nan

        sex = pd.to_numeric(df_da.get("sex"), errors="coerce")
        pop_s = float(len(df_da))
        male_s = float((sex == 1).sum()) if sex is not None else np.nan
        female_s = float((sex == 0).sum()) if sex is not None else np.nan
        hh_s = float(df_da.loc[df_da.get("HID", -1) != -1, "HID"].nunique()) if "HID" in df_da.columns else np.nan

        # Household-size buckets in synthetic (1..5+)
        hh_size_syn = {f"hh_{i}p_syn": 0.0 for i in [1, 2, 3, 4]}
        hh_size_syn["hh_5plus_syn"] = 0.0
        if "HID" in df_da.columns:
            sizes = df_da.loc[df_da["HID"] != -1].groupby("HID").size().clip(upper=5)
            hh_size_syn["hh_1p_syn"] = float((sizes == 1).sum())
            hh_size_syn["hh_2p_syn"] = float((sizes == 2).sum())
            hh_size_syn["hh_3p_syn"] = float((sizes == 3).sum())
            hh_size_syn["hh_4p_syn"] = float((sizes == 4).sum())
            hh_size_syn["hh_5plus_syn"] = float((sizes == 5).sum())

        hh_size_cen = {
            "hh_1p_cen": _sum_varids(census_da, [hhsize_vb[0]]) if 0 in hhsize_vb else np.nan,
            "hh_2p_cen": _sum_varids(census_da, [hhsize_vb[1]]) if 1 in hhsize_vb else np.nan,
            "hh_3p_cen": _sum_varids(census_da, [hhsize_vb[2]]) if 2 in hhsize_vb else np.nan,
            "hh_4p_cen": _sum_varids(census_da, [hhsize_vb[3]]) if 3 in hhsize_vb else np.nan,
            "hh_5plus_cen": _sum_varids(census_da, [hhsize_vb[4]]) if 4 in hhsize_vb else np.nan,
        }

        inc_syn = _syn_income_bins(df_da)
        inc_cen = _census_income_bins(census_da, totinc_vb)

        hht_syn = _syn_household_type_counts(df_da)
        hht_cen = _census_household_type_counts(census_da, cfstat_vb)

        row = {
            "da_code": str(da),
            "pop_syn": pop_s,
            "pop_cen": pop_c if pop_c is not None else np.nan,
            "pop_age_total_cen": pop_age_c if pop_age_c is not None else np.nan,
            "pop_abs_err": abs(pop_s - (pop_c if pop_c is not None else 0)),
            "pop_pct_err": 100.0 * _safe_div(abs(pop_s - (pop_c if pop_c is not None else 0)), pop_c if pop_c is not None else np.nan),
            "male_syn": male_s,
            "female_syn": female_s,
            "male_cen": male_c if male_c is not None else np.nan,
            "female_cen": female_c if female_c is not None else np.nan,
            "hh_syn": hh_s,
            "hh_cen": hh_c if hh_c is not None else np.nan,
            "hh_abs_err": abs(hh_s - (hh_c if hh_c is not None else 0)),
            "hh_pct_err": 100.0 * _safe_div(abs(hh_s - (hh_c if hh_c is not None else 0)), hh_c if hh_c is not None else np.nan),
        }
        row.update(hh_size_syn)
        row.update(hh_size_cen)
        row.update({f"{k}_syn": v for k, v in inc_syn.items()})
        row.update({f"{k}_cen": v for k, v in inc_cen.items()})
        row.update({f"{k}_syn": v for k, v in hht_syn.items()})
        row.update({f"{k}_cen": v for k, v in hht_cen.items()})
        rows.append(row)

    return pd.DataFrame(rows).sort_values("da_code").reset_index(drop=True)

