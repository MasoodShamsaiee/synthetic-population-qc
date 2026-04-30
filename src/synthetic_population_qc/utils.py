"""Shared low-level helpers used across preprocessing and evaluation code."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def norm_code(value) -> str | None:
    """Normalize geocodes/IDs to stripped string form while preserving nulls."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    try:
        return str(int(float(text)))
    except Exception:
        return text


def build_reference_census(
    census: pd.DataFrame,
    *,
    da_codes: Iterable[str],
    reference_geocode: str = "reference",
) -> pd.DataFrame:
    """Aggregate multiple DA rows into one synthetic reference geography row."""
    work = census.copy()
    if "geocode_norm" not in work.columns:
        work["geocode_norm"] = work["geocode"].map(norm_code)

    keep = {str(code) for code in da_codes}
    if not keep:
        return pd.DataFrame(columns=["geocode", "variable", "variableId", "total", "totalMale", "totalFemale"])

    ref = work.loc[work["geocode_norm"].isin(keep)].copy()
    if ref.empty:
        return pd.DataFrame(columns=["geocode", "variable", "variableId", "total", "totalMale", "totalFemale"])

    for col in ["total", "totalMale", "totalFemale"]:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")

    grouped = (
        ref.groupby(["variableId", "variable"], as_index=False)[["total", "totalMale", "totalFemale"]]
        .sum(min_count=1)
    )
    grouped["geocode"] = str(reference_geocode)
    grouped["geocode_norm"] = str(reference_geocode)
    return grouped[["geocode", "geocode_norm", "variable", "variableId", "total", "totalMale", "totalFemale"]]


def sample_label(sample_ratio: float | None) -> str:
    """Return a filesystem-friendly label for a DA sampling ratio."""
    return "all" if sample_ratio is None else f"sample{str(sample_ratio).replace('.', 'p')}"


def target_count(total_codes: int, sample_ratio: float | None) -> int:
    """Convert a sample ratio into a concrete DA count with sensible bounds."""
    if total_codes <= 0:
        return 0
    if sample_ratio is None:
        return int(total_codes)
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be in (0, 1] or None")
    return int(min(total_codes, max(1, round(total_codes * sample_ratio))))


def collect_checkpoint_logs(
    *,
    out_dir: str | Path,
    checkpoint_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and combine checkpoint metadata/failed-batch logs from one output directory."""
    out_dir = Path(out_dir)
    meta_frames: list[pd.DataFrame] = []
    failed_frames: list[pd.DataFrame] = []

    for path in sorted(out_dir.glob(f"{checkpoint_prefix}_meta_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            meta_frames.append(df)

    for path in sorted(out_dir.glob(f"{checkpoint_prefix}_failed_*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            failed_frames.append(df)

    meta_df = pd.concat(meta_frames, ignore_index=True) if meta_frames else pd.DataFrame()
    if not meta_df.empty and "batch_id" in meta_df.columns:
        meta_df = meta_df.sort_values(["batch_id", "from_idx", "to_idx"]).reset_index(drop=True)

    failed_df = pd.concat(failed_frames, ignore_index=True) if failed_frames else pd.DataFrame()
    if not failed_df.empty:
        sort_cols = [col for col in ["batch_id", "da_code", "error"] if col in failed_df.columns]
        if sort_cols:
            failed_df = failed_df.sort_values(sort_cols).reset_index(drop=True)

    return meta_df, failed_df
