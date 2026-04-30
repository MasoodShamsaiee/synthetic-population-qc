"""Compact reporting helpers for workflow fit summaries."""

from __future__ import annotations

import pandas as pd


def classify_fit(tvd: float | int | None) -> str:
    """Bucket a TVD score into a qualitative reporting label."""
    if tvd is None or pd.isna(tvd):
        return "not assessed"
    if float(tvd) < 0.02:
        return "strong"
    if float(tvd) < 0.10:
        return "good"
    if float(tvd) < 0.25:
        return "moderate"
    return "weak"


def build_results_summary(aligned_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a concise attribute-level table for reporting workflow fit."""
    if aligned_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "unit",
                "attribute",
                "tvd",
                "mae_pp",
                "max_abs_pp",
                "seed_to_census_ratio",
                "fit_rating",
                "recommended_use",
            ]
        )

    out = aligned_summary_df.copy()
    if "seed_to_census_ratio" in out.columns and "seed_to_reference_ratio" in out.columns:
        out["seed_to_census_ratio"] = out["seed_to_census_ratio"].where(
            out["seed_to_census_ratio"].notna(),
            out["seed_to_reference_ratio"],
        )
    out["fit_rating"] = out["tvd"].apply(classify_fit)

    def _takeaway(row: pd.Series) -> str:
        attr = str(row["attribute"])
        rating = str(row["fit_rating"])
        if rating == "strong":
            return f"{attr} is well aligned and ready to report as a strong result."
        if rating == "good":
            return f"{attr} is reasonably aligned and suitable for the main results with light caveats."
        if rating == "moderate":
            return f"{attr} is usable, but should be reported with caution and discussed as a partial-strength result."
        if rating == "weak":
            return f"{attr} is currently weak and should be framed as a limitation or refinement target."
        return f"{attr} has not yet been assessed."

    out["recommended_use"] = out.apply(_takeaway, axis=1)
    keep_cols = [
        "unit",
        "attribute",
        "tvd",
        "mae_pp",
        "max_abs_pp",
        "seed_to_census_ratio",
        "fit_rating",
        "recommended_use",
    ]
    return out[keep_cols].sort_values(["unit", "tvd", "attribute"]).reset_index(drop=True)
