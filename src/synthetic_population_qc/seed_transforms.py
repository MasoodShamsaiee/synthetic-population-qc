"""Seed harmonization transforms for PUMF-coded demographic variables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_age_grp(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw `agegrp` codes into the harmonized workflow age buckets."""
    out = df.copy()
    for value in range(17, 22):
        out.loc[out["agegrp"] == value, "agegrp"] = value + 8
    for value in range(16, 7, -1):
        out.loc[out["agegrp"] == value, "agegrp"] = value + 7
    out.loc[out["agegrp"] == 1, "agegrp"] = 10
    out.loc[out["agegrp"] == 2, "agegrp"] = 11
    out.loc[out["agegrp"] == 3, "agegrp"] = 11
    out.loc[out["agegrp"] == 4, "agegrp"] = 12
    out.loc[out["agegrp"] == 5, "agegrp"] = 12
    out.loc[out["agegrp"] == 6, "agegrp"] = 14
    out.loc[out["agegrp"] == 7, "agegrp"] = 14
    # The public workflow now keeps one final 85+ age bucket because the DA targets
    # do not publish separate 85-89 / 90-94 / 95-99 / 100+ marginals.
    out.loc[pd.to_numeric(out["agegrp"], errors="coerce").isin([27, 28, 29]), "agegrp"] = 26
    return out.loc[out["agegrp"] != 88].copy()


def map_hdgree(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse detailed education codes into the three workflow groups."""
    out = df.copy()
    out.loc[out["hdgree"] == 88, "hdgree"] = 1
    out.loc[out["hdgree"] == 99, "hdgree"] = 1
    out.loc[out["hdgree"] > 2, "hdgree"] = 1686
    out.loc[out["hdgree"] == 1, "hdgree"] = 1684
    out.loc[out["hdgree"] == 2, "hdgree"] = 1685
    return out


def map_lfact(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse labour-force detail codes into the workflow status groups."""
    out = df.copy()
    out.loc[out["lfact"] == 1, "lfact"] = 1867
    out.loc[out["lfact"] == 2, "lfact"] = 1867
    out.loc[out["lfact"] < 11, "lfact"] = 1868
    out.loc[out["lfact"] < 100, "lfact"] = 1869
    return out


def map_hhsize(df: pd.DataFrame) -> pd.DataFrame:
    """Cap household size at the workflow maximum bucket."""
    out = df.copy()
    out.loc[out["hhsize"] == 8, "hhsize"] = 1
    out.loc[out["hhsize"] > 5, "hhsize"] = 5
    return out


def map_totinc(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw total-income values into the workflow income buckets."""
    out = df.loc[df["TotInc"] != 88888888].copy()
    out.loc[out["TotInc"] == 99999999, "TotInc"] = 695
    out.loc[out["TotInc"] < 20000, "TotInc"] = 695
    out.loc[(out["TotInc"] >= 20000) & (out["TotInc"] < 60000), "TotInc"] = 697
    out.loc[(out["TotInc"] >= 60000) & (out["TotInc"] < 100000), "TotInc"] = 701
    out.loc[out["TotInc"] >= 100000, "TotInc"] = 705
    return out


def map_prihm(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the private-household-maintainer indicator."""
    out = df.copy()
    out.loc[out["prihm"] == 9, "prihm"] = 0
    return out


def map_cfstat(df: pd.DataFrame, *, src_col: str = "cfstat", dst_col: str = "cfstat") -> pd.DataFrame:
    """Collapse raw family-status codes into the harmonized workflow values."""
    out = df.copy()
    if dst_col != src_col:
        out[dst_col] = out[src_col]
    mapping = {
        1: 0,
        2: 1,
        4: 1,
        3: 2,
        5: 2,
        7: 3,
        8: 3,
        6: 4,
    }
    out[dst_col] = out[dst_col].map(mapping)
    return out


def probabilistic_sampling(probabilities, total_pop: int) -> np.ndarray:
    """Sample an integer contingency table from a fitted fractional distribution."""
    if isinstance(probabilities, dict):
        result_arr = probabilities["result"]
    elif isinstance(probabilities, tuple):
        result_arr = probabilities[0]
    elif isinstance(probabilities, np.ndarray):
        result_arr = probabilities
    else:
        raise TypeError(f"Unsupported type for probabilities: {type(probabilities)}")

    probas = np.asarray(result_arr, dtype=np.float64).ravel()
    total_mass = probas.sum()
    if total_mass <= 0:
        raise ValueError("Result array has zero total mass; cannot sample.")
    probas /= total_mass
    draws = np.random.multinomial(int(total_pop), probas)
    return draws.reshape(result_arr.shape).astype(np.int32, copy=False)
