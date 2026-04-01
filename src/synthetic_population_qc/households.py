from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class HouseholdAssignmentSummary:
    da_code: str
    n_people: int
    target_households: int
    assigned_households: int
    leftover_people: int


def age_from_agegrp(agegrp: pd.Series) -> pd.Series:
    """
    Approximate age (years) from synthetic agegrp index used in the copied pipeline.

    Expected bins:
    0..16 -> 5-year bins [0-4, 5-9, ..., 80-84]
    17    -> 85+
    """
    x = pd.to_numeric(agegrp, errors="coerce")
    out = pd.Series(np.nan, index=agegrp.index, dtype=float)
    mask = x.notna()
    # midpoint for 5-year bins
    out.loc[mask] = x.loc[mask] * 5 + 2
    # top-coded 85+
    out.loc[x == 17] = 87
    return out


def compute_hhtypes(df_pop: pd.DataFrame) -> pd.DataFrame:
    """
    Classify household type by household composition.

    Output codes:
      0 = couple without children
      1 = couple with children
      2 = one-parent family
      3 = one-person household
      4 = other
    """
    out = df_pop.copy()
    out["hhtype"] = -1

    if "HID" not in out.columns:
        raise KeyError("Expected 'HID' column in df_pop.")
    if "age" not in out.columns:
        raise KeyError("Expected 'age' column in df_pop.")

    valid_hids = out.loc[out["HID"] != -1, "HID"].dropna().unique()
    for hh_id in valid_hids:
        hh = out.loc[out["HID"] == hh_id]
        n = len(hh)
        ages = sorted(pd.to_numeric(hh["age"], errors="coerce").dropna().tolist())
        if not ages:
            continue

        if n == 1:
            out.loc[out["HID"] == hh_id, "hhtype"] = 3
        elif n == 2:
            if abs(ages[0] - ages[1]) > 16:
                out.loc[out["HID"] == hh_id, "hhtype"] = 2
            elif ages[0] > 16 and ages[1] > 16:
                out.loc[out["HID"] == hh_id, "hhtype"] = 0
        elif n >= 3:
            n_children = sum(a < 16 for a in ages)
            n_adults = sum(a >= 16 for a in ages)
            if n_adults >= 2 and n_children >= 1:
                out.loc[out["HID"] == hh_id, "hhtype"] = 1
            elif n_adults == 1 and n_children >= 1:
                out.loc[out["HID"] == hh_id, "hhtype"] = 2

    out.loc[(out["HID"] != -1) & (out["hhtype"] == -1), "hhtype"] = 4
    return out


def _safe_int(v) -> int | None:
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except Exception:
        return None


def _total_for_var(df: pd.DataFrame, var_id: int) -> int | None:
    s = df.loc[df["variableId"] == var_id, "total"]
    if s.empty:
        return None
    v = s.iloc[0]
    if v in ("x", "F", ".."):
        return None
    return _safe_int(v)


def _household_size_slots_from_census(
    da_census: pd.DataFrame,
    province_census: pd.DataFrame | None,
    total_hh_vb_id: int,
    hhsize_vb: dict[int, int],
    *,
    size_5plus: int = 5,
) -> list[int]:
    """
    Build a list of household-size slots for a DA from census household counts.
    """
    target_hh = _total_for_var(da_census, total_hh_vb_id)
    if target_hh is None:
        # No valid DA households; fallback will happen in caller.
        return []

    counts: dict[int, int] = {}
    for k in sorted(hhsize_vb.keys()):
        size = k + 1 if k < 4 else int(size_5plus)
        vv = _total_for_var(da_census, hhsize_vb[k])
        if vv is None and province_census is not None:
            # Province fallback by share.
            pv = _total_for_var(province_census, hhsize_vb[k])
            pt = _total_for_var(province_census, total_hh_vb_id)
            if pv is not None and pt and pt > 0:
                vv = int(round(target_hh * (pv / pt)))
        counts[size] = max(0, int(vv or 0))

    # Reconcile to exact total households.
    cur = sum(counts.values())
    if cur != target_hh:
        # Adjust the most common bin (or size 2 by default).
        key = max(counts, key=counts.get) if counts else 2
        counts[key] = max(0, counts.get(key, 0) + (target_hh - cur))

    slots: list[int] = []
    for size, n in sorted(counts.items()):
        slots.extend([int(size)] * int(n))
    return slots


def _sample_without_replacement(rng: np.random.Generator, pool: list[int], n: int) -> list[int]:
    if n <= 0 or len(pool) == 0:
        return []
    n = min(n, len(pool))
    picked = rng.choice(np.asarray(pool, dtype=int), size=n, replace=False)
    return [int(x) for x in picked.tolist()]


def assign_households_for_da(
    df_da: pd.DataFrame,
    hh_slots: Iterable[int],
    *,
    area_col: str = "area",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Assign household IDs to one DA using precomputed household-size slots.
    """
    rng = np.random.default_rng(int(random_seed))
    out = df_da.copy()
    out["HID"] = -1

    if "age" not in out.columns:
        if "agegrp" not in out.columns:
            raise KeyError("Need either 'age' or 'agegrp' to assign households.")
        out["age"] = age_from_agegrp(out["agegrp"]).round().astype("Int64")

    out["_adult"] = pd.to_numeric(out["age"], errors="coerce").fillna(0) >= 18
    out["_child"] = ~out["_adult"]

    remaining = set(out.index.tolist())
    hid = 0

    # Prefer larger households first to reduce fragmentation.
    slots = sorted([int(max(1, s)) for s in hh_slots], reverse=True)
    for size in slots:
        if not remaining:
            break

        # Candidate pools.
        rem_list = list(remaining)
        adults = [i for i in rem_list if bool(out.at[i, "_adult"])]
        children = [i for i in rem_list if bool(out.at[i, "_child"])]

        members: list[int] = []
        if size == 1:
            # Prefer one-person profile if cfstat exists.
            if "cfstat" in out.columns:
                onep = [i for i in rem_list if _safe_int(out.at[i, "cfstat"]) == 4]
                members = _sample_without_replacement(rng, onep or rem_list, 1)
            else:
                members = _sample_without_replacement(rng, rem_list, 1)
        else:
            # Default family-like composition:
            # >=3: try 2 adults + children
            # 2 : try 2 adults, else 1 adult + 1 child, else any 2
            if size >= 3:
                members.extend(_sample_without_replacement(rng, adults, 2))
                need = size - len(members)
                if need > 0:
                    members.extend(_sample_without_replacement(rng, children, need))
                need = size - len(members)
                if need > 0:
                    rem_after = [i for i in rem_list if i not in members]
                    members.extend(_sample_without_replacement(rng, rem_after, need))
            else:
                two_adults = _sample_without_replacement(rng, adults, 2)
                if len(two_adults) == 2:
                    members = two_adults
                else:
                    members = _sample_without_replacement(rng, rem_list, 2)

        if not members:
            continue

        out.loc[members, "HID"] = hid
        hid += 1
        for i in members:
            remaining.discard(i)

    # Assign any leftovers as singleton households.
    for i in list(remaining):
        out.at[i, "HID"] = hid
        hid += 1

    out["HID"] = out["HID"].astype(int)
    out["area"] = out[area_col].astype(str)
    out = out.drop(columns=["_adult", "_child"], errors="ignore")
    out = compute_hhtypes(out)
    return out


def assign_households_using_census(
    syn_inds: pd.DataFrame,
    census: pd.DataFrame,
    *,
    province: str = "24",
    area_col: str = "area",
    total_hh_vb_id: int,
    hhsize_vb: dict[int, int],
    random_seed: int = 42,
    size_5plus: int = 5,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign HID + hhtype to synthetic individuals DA-by-DA using census household-size slots.

    Returns
    -------
    (syn_with_hh, summary_df)
    """
    from tqdm.auto import tqdm

    if area_col not in syn_inds.columns:
        raise KeyError(f"Expected '{area_col}' column in syn_inds.")

    work = syn_inds.copy()
    work[area_col] = work[area_col].astype(str)

    prov_census = census.loc[census["geocode"].astype(str) == str(province)].copy()
    by_da = work.groupby(area_col, sort=True)

    parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    iterator = tqdm(by_da, desc="Assigning households") if show_progress else by_da
    for da_code, g in iterator:
        da_census = census.loc[census["geocode"].astype(str) == str(da_code)].copy()
        slots = _household_size_slots_from_census(
            da_census=da_census,
            province_census=prov_census,
            total_hh_vb_id=int(total_hh_vb_id),
            hhsize_vb=hhsize_vb,
            size_5plus=size_5plus,
        )

        if not slots:
            # fallback: use each person's hhsize field as slot target if present
            if "hhsize" in g.columns:
                hh = pd.to_numeric(g["hhsize"], errors="coerce").fillna(1).astype(int).clip(lower=1)
                slots = hh.tolist()
            else:
                slots = [1] * len(g)

        assigned = assign_households_for_da(
            g,
            hh_slots=slots,
            area_col=area_col,
            random_seed=random_seed + int(abs(hash(str(da_code))) % 10_000_000),
        )
        parts.append(assigned)

        n_people = len(g)
        target_hh = len(slots)
        got_hh = int(assigned["HID"].nunique())
        leftover = max(0, n_people - sum(slots))
        summary_rows.append(
            HouseholdAssignmentSummary(
                da_code=str(da_code),
                n_people=int(n_people),
                target_households=int(target_hh),
                assigned_households=int(got_hh),
                leftover_people=int(leftover),
            ).__dict__
        )

    out = pd.concat(parts, axis=0, ignore_index=True) if parts else work.copy()
    summary_df = pd.DataFrame(summary_rows)
    return out, summary_df

