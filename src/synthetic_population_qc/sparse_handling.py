"""Fallback assignment helpers for attributes with weak conditional support."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def conditional_support_weight(
    seed_df: pd.DataFrame,
    *,
    attr: str,
    cond_cols: list[str],
    weight_col: str = "weight",
) -> float:
    """Measure the weakest non-zero conditional cell for one attribute."""
    cols = list(cond_cols) + [attr, weight_col]
    if any(col not in seed_df.columns for col in cols):
        return 0.0
    work = seed_df[cols].dropna(subset=[attr]).copy()
    if cond_cols:
        work = work.dropna(subset=cond_cols)
    if work.empty:
        return 0.0
    grouped = work.groupby(list(cond_cols) + [attr], dropna=False)[weight_col].sum().astype(float)
    grouped = grouped[grouped > 0]
    return float(grouped.min()) if not grouped.empty else 0.0


def _conditional_prob_table(
    seed_df: pd.DataFrame,
    *,
    attr: str,
    cond_cols: list[str],
    weight_col: str = "weight",
) -> dict[tuple, dict[str, float]]:
    """Estimate conditional category probabilities from the weighted seed."""
    work = seed_df[cond_cols + [attr, weight_col]].dropna(subset=[attr]).copy()
    if cond_cols:
        work = work.dropna(subset=cond_cols)
    grouped = work.groupby(cond_cols + [attr], dropna=False)[weight_col].sum().reset_index()
    result: dict[tuple, dict[str, float]] = {}
    if cond_cols:
        for cond_vals, part in grouped.groupby(cond_cols, dropna=False, sort=False):
            if not isinstance(cond_vals, tuple):
                cond_vals = (cond_vals,)
            weights = part.set_index(attr)[weight_col].astype(float)
            total = float(weights.sum())
            result[tuple(cond_vals)] = (weights / total).to_dict() if total > 0 else {}
    else:
        weights = grouped.set_index(attr)[weight_col].astype(float)
        total = float(weights.sum())
        result[tuple()] = (weights / total).to_dict() if total > 0 else {}
    return result


def _global_probabilities(seed_df: pd.DataFrame, *, attr: str, weight_col: str = "weight") -> dict[str, float]:
    """Compute unconditional category probabilities for the fallback tier."""
    work = seed_df[[attr, weight_col]].dropna(subset=[attr]).copy()
    counts = work.groupby(attr, dropna=False)[weight_col].sum().astype(float)
    total = float(counts.sum())
    return (counts / total).to_dict() if total > 0 else {}


def _ipf_fit(seed: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, n_iter: int = 80) -> np.ndarray:
    """Fit a dense seed matrix to row/column targets with simple two-way IPF."""
    arr = np.asarray(seed, dtype=float).copy()
    arr = np.clip(arr, 1e-12, None)
    for _ in range(n_iter):
        row_sums = arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        arr *= row_targets[:, None] / row_sums
        col_sums = arr.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        arr *= col_targets[None, :] / col_sums
    return arr


def _integerize_matrix(arr: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray) -> np.ndarray:
    """Convert a fractional IPF table into integer counts via largest remainder."""
    floors = np.floor(arr).astype(int)
    row_need = row_targets.astype(int) - floors.sum(axis=1)
    col_need = col_targets.astype(int) - floors.sum(axis=0)
    frac = arr - floors
    candidates = sorted(
        ((float(frac[i, j]), i, j) for i in range(arr.shape[0]) for j in range(arr.shape[1])),
        reverse=True,
    )
    for _, i, j in candidates:
        if row_need[i] <= 0 or col_need[j] <= 0:
            continue
        floors[i, j] += 1
        row_need[i] -= 1
        col_need[j] -= 1
    return floors


def _allocate_counts_by_group(
    *,
    group_keys: list[tuple],
    group_sizes: list[int],
    categories: list[str],
    target_counts: dict[str, int],
    cond_probs: dict[tuple, dict[str, float]],
    fallback_probs: dict[str, float],
) -> dict[tuple, dict[str, int]]:
    """Allocate category totals across conditioning groups with IPF."""
    if not group_keys:
        return {}
    seed = np.zeros((len(group_keys), len(categories)), dtype=float)
    for i, key in enumerate(group_keys):
        probs = cond_probs.get(key) or fallback_probs
        for j, cat in enumerate(categories):
            seed[i, j] = float(probs.get(cat, 0.0))
        if seed[i].sum() <= 0:
            seed[i, :] = 1.0
    row_targets = np.asarray(group_sizes, dtype=float)
    col_targets = np.asarray([int(target_counts.get(cat, 0)) for cat in categories], dtype=float)
    fitted = _ipf_fit(seed, row_targets=row_targets, col_targets=col_targets)
    ints = _integerize_matrix(fitted, row_targets=row_targets, col_targets=col_targets)
    return {
        key: {categories[j]: int(ints[i, j]) for j in range(len(categories))}
        for i, key in enumerate(group_keys)
    }


def _deterministic_attribute_assignment(
    df: pd.DataFrame,
    *,
    seed_df: pd.DataFrame,
    attr: str,
    cond_cols: list[str],
    target_counts: dict[str, int],
    row_id_col: str,
    weight_col: str = "weight",
) -> pd.Series:
    """Assign categories deterministically after group-level counts are allocated."""
    work = df.copy()
    categories = [cat for cat, n in target_counts.items() if int(n) > 0]
    if not categories:
        return pd.Series(pd.NA, index=work.index, dtype="object", name=attr)
    grouped = work.groupby(cond_cols, dropna=False, sort=False) if cond_cols else [(tuple(), work)]
    group_keys: list[tuple] = []
    group_sizes: list[int] = []
    group_frames: dict[tuple, pd.DataFrame] = {}
    for key, part in grouped:
        key = key if isinstance(key, tuple) else (key,)
        group_keys.append(key)
        group_sizes.append(int(len(part)))
        group_frames[key] = part.sort_values(row_id_col)
    cond_probs = _conditional_prob_table(seed_df, attr=attr, cond_cols=cond_cols, weight_col=weight_col)
    fallback_probs = _global_probabilities(seed_df, attr=attr, weight_col=weight_col)
    allocation = _allocate_counts_by_group(
        group_keys=group_keys,
        group_sizes=group_sizes,
        categories=categories,
        target_counts=target_counts,
        cond_probs=cond_probs,
        fallback_probs=fallback_probs,
    )
    assigned = pd.Series(index=work.index, dtype="object")
    for key in group_keys:
        part = group_frames[key]
        start = 0
        # Rows are pre-sorted by the stable identifier so assignment is
        # repeatable across runs without adding extra random state here.
        for cat in categories:
            n = int(allocation[key].get(cat, 0))
            if n <= 0:
                continue
            rows = part.index[start : start + n]
            assigned.loc[rows] = cat
            start += n
    if assigned.isna().any():
        remaining = assigned[assigned.isna()].index
        fallback = max(target_counts, key=lambda k: target_counts[k]) if target_counts else pd.NA
        assigned.loc[remaining] = fallback
    return assigned.rename(attr)


def assign_attribute_with_fallback(
    df: pd.DataFrame,
    *,
    seed_df: pd.DataFrame,
    attr: str,
    target_counts: dict[str, int],
    row_id_col: str,
    fallback_ladder: list[list[str]] | list[tuple[str, ...]] | None,
    min_conditional_weight: float = 10.0,
    weight_col: str = "weight",
) -> tuple[pd.Series, dict[str, object]]:
    """Choose the strongest viable conditioning tier, then assign deterministically."""
    ladders = [list(x) for x in (fallback_ladder or [])]
    if not ladders:
        ladders = [[]]
    chosen = []
    chosen_rank = len(ladders) - 1
    observed_support = 0.0
    for idx, cond_cols in enumerate(ladders):
        support = conditional_support_weight(seed_df, attr=attr, cond_cols=cond_cols, weight_col=weight_col)
        observed_support = max(observed_support, support)
        if support >= float(min_conditional_weight) or not cond_cols:
            chosen = cond_cols
            chosen_rank = idx
            break
    assigned = _deterministic_attribute_assignment(
        df,
        seed_df=seed_df,
        attr=attr,
        cond_cols=chosen,
        target_counts=target_counts,
        row_id_col=row_id_col,
        weight_col=weight_col,
    )
    report = {
        "attribute": attr,
        "chosen_conditioning_json": json.dumps(chosen),
        "fallback_rank": chosen_rank,
        "used_global_fallback": len(chosen) == 0,
        "used_category_aggregation": False,
        "selected_min_conditional_weight": conditional_support_weight(
            seed_df, attr=attr, cond_cols=chosen, weight_col=weight_col
        ),
        "max_observed_min_conditional_weight": observed_support,
    }
    return assigned, report
