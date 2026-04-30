"""Comparison helpers for contrasting workflow outputs with DA census context."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from synthetic_population_qc.enrichment import HOUSEHOLD_ATTR_LABELS, PERSON_ATTR_LABELS
from synthetic_population_qc.energy_workflow import HOUSEHOLD_COLLAPSE_MAP, PERSON_COLLAPSE_MAP


COMPARISON_SPECS = {
    "dwelling_type": {
        "unit": "household",
        "table_name": "dwelling_characteristics",
        "label_map": HOUSEHOLD_ATTR_LABELS["dwelling_type"],
        "collapse_map": HOUSEHOLD_COLLAPSE_MAP["dwelling_type"],
        "synth_col": "dwelling_type",
        "denominator": "strict_households",
    },
    "tenure": {
        "unit": "household",
        "table_name": "housing",
        "label_map": HOUSEHOLD_ATTR_LABELS["tenure"],
        "collapse_map": HOUSEHOLD_COLLAPSE_MAP["tenure"],
        "synth_col": "tenure",
        "denominator": "strict_households",
    },
    "period_built": {
        "unit": "household",
        "table_name": "housing",
        "label_map": HOUSEHOLD_ATTR_LABELS["period_built"],
        "collapse_map": HOUSEHOLD_COLLAPSE_MAP["period_built"],
        "synth_col": "period_built",
        "denominator": "strict_households",
    },
    "citizenship_status": {
        "unit": "person",
        "table_name": "immigration_citizenship",
        "label_map": PERSON_ATTR_LABELS["citizenship_status"],
        "collapse_map": PERSON_COLLAPSE_MAP["citizenship_status"],
        "synth_col": "citizenship_status",
        "denominator": "strict_people",
    },
    "immigrant_status": {
        "unit": "person",
        "table_name": "immigration_citizenship",
        "label_map": PERSON_ATTR_LABELS["immigrant_status"],
        "collapse_map": PERSON_COLLAPSE_MAP["immigrant_status"],
        "synth_col": "immigrant_status",
        "denominator": "strict_people",
    },
    "commute_mode": {
        "unit": "person",
        "table_name": "commute",
        "label_map": PERSON_ATTR_LABELS["commute_mode"],
        "collapse_map": PERSON_COLLAPSE_MAP["commute_mode"],
        "synth_col": "commute_mode",
        "denominator": "restricted_universe",
    },
}


PRESENTATION_ATTRIBUTE_LABELS = {
    "tenure": "Tenure",
    "dwelling_type": "Dwelling Type",
    "period_built": "Period Built",
    "citizenship_status": "Citizenship Status",
    "commute_mode": "Commute Mode",
}


PRESENTATION_CATEGORY_LABELS = {
    "renter_or_band": "renter",
    "active_transport": "walk_or_bike",
    "single_detached_house": "single_detached_house",
    "apartment": "apartment",
    "other_dwelling": "other_ground_or_attached",
}


CATEGORY_EXPLANATIONS = {
    "renter": "Renter households; the original combined label also included band housing, but that distinction is not meaningful for this Montreal subset.",
    "walk_or_bike": "Commuting by walking or bicycle.",
    "single_detached_house": "A detached house intended for one household.",
    "apartment": "Combined apartment category: apartment or flat in a duplex, apartment in a low-rise building, and apartment in a high-rise building.",
    "other_ground_or_attached": "Combined non-apartment attached/other category: semi-detached, row house, other single-attached, and movable dwellings.",
}


def _selected_da_codes(people_df: pd.DataFrame, households_df: pd.DataFrame) -> list[str]:
    """Collect the sorted DA codes present in the current synthetic outputs."""
    selected_codes: set[str] = set()
    if "area" in people_df.columns:
        selected_codes |= set(people_df["area"].dropna().astype(str).str.strip())
    if "area" in households_df.columns:
        selected_codes |= set(households_df["area"].dropna().astype(str).str.strip())
    return sorted(code for code in selected_codes if code)


def build_census_attr_frame(
    attr: str,
    *,
    context_tables: dict[str, pd.DataFrame],
    selected_da_codes: list[str],
) -> pd.DataFrame:
    """Build one long-form census frame for a harmonized comparison attribute."""
    spec = COMPARISON_SPECS[attr]
    table = context_tables.get(spec["table_name"], pd.DataFrame()).copy()
    if table.empty or "da_code" not in table.columns:
        return pd.DataFrame(columns=["da_code", "category", "count", "share", "source", "attribute"])

    table["da_code"] = table["da_code"].astype(str).str.strip()
    if selected_da_codes:
        table = table.loc[table["da_code"].isin(selected_da_codes)].copy()

    rows: list[dict[str, object]] = []
    for _, row in table.iterrows():
        values: dict[str, float] = {}
        for out_key, in_keys in spec["collapse_map"].items():
            total = 0.0
            for in_key in in_keys:
                col = spec["label_map"].get(in_key)
                if col is not None:
                    total += float(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").fillna(0.0).iloc[0])
            values[out_key] = total
        denom = float(sum(values.values()))
        for category, count in values.items():
            rows.append(
                {
                    "da_code": row["da_code"],
                    "category": category,
                    "count": float(count),
                    "share": (float(count) / denom) if denom > 0 else None,
                    "source": "census",
                    "attribute": attr,
                }
            )
    return pd.DataFrame(rows)


def build_synth_attr_frame(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one long-form synthetic frame for a harmonized comparison attribute."""
    spec = COMPARISON_SPECS[attr]
    if spec["unit"] == "household":
        df = households_df[["area", spec["synth_col"]]].dropna().copy()
    else:
        df = people_df[["area", spec["synth_col"]]].dropna().copy()
    df = df.rename(columns={"area": "da_code", spec["synth_col"]: "category"})
    if df.empty:
        return pd.DataFrame(columns=["da_code", "category", "count", "share", "source", "attribute"])
    counts = df.value_counts(["da_code", "category"]).rename("count").reset_index()
    all_categories = list(spec["collapse_map"].keys())
    full_index = pd.MultiIndex.from_product(
        [counts["da_code"].drop_duplicates(), all_categories],
        names=["da_code", "category"],
    )
    counts = (
        counts.set_index(["da_code", "category"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    counts["share"] = counts["count"] / counts.groupby("da_code")["count"].transform("sum")
    counts["source"] = "synthetic"
    counts["attribute"] = attr
    counts["da_code"] = counts["da_code"].astype(str).str.strip()
    return counts


def build_comparison_diagnostics(
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize whether each comparison attribute is valid for this bundle."""
    selected_codes = _selected_da_codes(people_df, households_df)
    synth_people_total = float(len(people_df))
    synth_household_total = float(len(households_df))

    rows: list[dict[str, object]] = []
    for attr, spec in COMPARISON_SPECS.items():
        census = build_census_attr_frame(attr, context_tables=context_tables, selected_da_codes=selected_codes)
        synth = build_synth_attr_frame(attr, people_df=people_df, households_df=households_df)

        census_total = float(census.groupby("da_code", dropna=False)["count"].sum().sum()) if not census.empty else 0.0
        synth_total = float(synth.groupby("da_code", dropna=False)["count"].sum().sum()) if not synth.empty else 0.0
        expected_total = synth_household_total if spec["unit"] == "household" else synth_people_total
        matched_das = int(census["da_code"].nunique()) if not census.empty else 0
        selected_da_count = int(len(selected_codes))
        coverage_share = (matched_das / selected_da_count) if selected_da_count > 0 else 0.0

        denominator_ratio = (synth_total / census_total) if census_total > 0 else None
        comparison_valid = True
        warning = None

        if matched_das == 0:
            comparison_valid = False
            warning = "No matching DA rows in filtered census context table."
        elif coverage_share < 0.95:
            comparison_valid = False
            warning = f"Low DA coverage in census table ({coverage_share:.1%})."
        elif spec["denominator"] != "restricted_universe" and census_total > 0:
            ratio_to_expected = expected_total / census_total
            if ratio_to_expected < 0.5 or ratio_to_expected > 2.0:
                comparison_valid = False
                warning = (
                    "Census total is on an incompatible denominator for this subset "
                    f"(expected_total/census_total={ratio_to_expected:.2f})."
                )

        rows.append(
            {
                "attribute": attr,
                "unit": spec["unit"],
                "table_name": spec["table_name"],
                "selected_da_count": selected_da_count,
                "matched_da_count": matched_das,
                "coverage_share": coverage_share,
                "census_total": census_total,
                "synthetic_total": synth_total,
                "expected_total": expected_total,
                "synthetic_to_census_ratio": denominator_ratio,
                "comparison_valid": comparison_valid,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows).sort_values(["comparison_valid", "attribute"], ascending=[True, True]).reset_index(drop=True)


def _ensure_valid(attr: str, diagnostics_df: pd.DataFrame) -> None:
    """Raise a clear error when a requested comparison is not valid."""
    match = diagnostics_df.loc[diagnostics_df["attribute"] == attr]
    if match.empty:
        raise ValueError(f"No diagnostics available for comparison attribute {attr!r}.")
    if not bool(match["comparison_valid"].iloc[0]):
        warning = match["warning"].iloc[0]
        raise ValueError(f"Comparison for {attr!r} is not valid for this filtered bundle. {warning}")


def _present_attr(attr: str) -> str:
    """Return the presentation label for a comparison attribute."""
    return PRESENTATION_ATTRIBUTE_LABELS.get(attr, attr.replace("_", " ").title())


def _present_category(category: str) -> str:
    """Return the presentation label for a comparison category."""
    return PRESENTATION_CATEGORY_LABELS.get(category, category)


def build_attr_comparison(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine census and synthetic shares for one comparison attribute."""
    _ensure_valid(attr, diagnostics_df)
    selected_codes = _selected_da_codes(people_df, households_df)
    return pd.concat(
        [
            build_census_attr_frame(attr, context_tables=context_tables, selected_da_codes=selected_codes),
            build_synth_attr_frame(attr, people_df=people_df, households_df=households_df),
        ],
        ignore_index=True,
    )


def build_overall_comparison(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate one comparison attribute across the full selected DA set."""
    comp = build_attr_comparison(
        attr,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    overall = comp.groupby(["source", "attribute", "category"], dropna=False)["count"].sum().reset_index()
    overall["share"] = overall["count"] / overall.groupby(["source", "attribute"])["count"].transform("sum")
    overall["attribute_label"] = overall["attribute"].map(_present_attr)
    overall["category_label"] = overall["category"].map(_present_category)
    overall["source_label"] = overall["source"].map({"census": "Census", "synthetic": "Synthetic"})
    return overall.sort_values(["source", "category"]).reset_index(drop=True)


def plot_overall_comparison(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
):
    """Plot the overall census-vs-synthetic comparison for one attribute."""
    overall = build_overall_comparison(
        attr,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    return px.bar(
        overall,
        x="category_label",
        y="share",
        color="source_label",
        barmode="group",
        title=f"{_present_attr(attr)} - Census vs Synthetic",
        hover_data=["count"],
        labels={"category_label": "category", "source_label": "source"},
    )


def build_da_share_distribution(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return DA-level share rows with presentation labels for plotting."""
    comp = build_attr_comparison(
        attr,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    out = comp.copy()
    out["attribute_label"] = out["attribute"].map(_present_attr)
    out["category_label"] = out["category"].map(_present_category)
    out["source_label"] = out["source"].map({"census": "Census", "synthetic": "Synthetic"})
    return out


def plot_split_violin_comparison(
    attr: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
):
    """Plot DA-level share distributions for census and synthetic values."""
    dist = build_da_share_distribution(
        attr,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    category_order = list(dist["category_label"].drop_duplicates())
    fig = go.Figure()
    for source_label, side, color in [
        ("Census", "negative", "#1f77b4"),
        ("Synthetic", "positive", "#d62728"),
    ]:
        subset = dist.loc[dist["source_label"] == source_label].copy()
        fig.add_trace(
            go.Violin(
                x=subset["category_label"],
                y=subset["share"],
                name=source_label,
                legendgroup=source_label,
                scalegroup=_present_attr(attr),
                side=side,
                line_color=color,
                meanline_visible=True,
                points=False,
                spanmode="hard",
                width=0.9,
                hovertemplate=(
                    "source=%{fullData.name}<br>"
                    "category=%{x}<br>"
                    "DA share=%{y:.3f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=f"DA Share Distribution: {_present_attr(attr)}",
        violingap=0.15,
        violinmode="overlay",
        xaxis_title="category",
        yaxis_title="share within DA",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=category_order)
    return fig


def build_comparison_glossary() -> pd.DataFrame:
    """Return a compact glossary for presentation-category labels."""
    return pd.DataFrame(
        [{"category": category, "meaning": meaning} for category, meaning in CATEGORY_EXPLANATIONS.items()]
    )


def build_da_share_comparison(
    attr: str,
    category: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare one category's census and synthetic shares by DA."""
    comp = build_attr_comparison(
        attr,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    census = comp.loc[comp["source"] == "census"].copy()
    synth = comp.loc[comp["source"] == "synthetic"].copy()
    census_share = census.loc[census["category"] == category, ["da_code", "share"]].rename(columns={"share": "census_share"})
    synth_share = synth.loc[synth["category"] == category, ["da_code", "share"]].rename(columns={"share": "synthetic_share"})
    out = census_share.merge(synth_share, on="da_code", how="outer").fillna({"census_share": 0.0, "synthetic_share": 0.0})
    out["share_diff"] = out["synthetic_share"] - out["census_share"]
    out["category_label"] = _present_category(category)
    return out


def build_da_comparison_map(
    attr: str,
    category: str,
    value_col: str,
    title: str,
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
    city_da_gdf: gpd.GeoDataFrame,
):
    """Join one DA-level comparison metric onto a DA geometry table."""
    comp = build_da_share_comparison(
        attr,
        category,
        people_df=people_df,
        households_df=households_df,
        context_tables=context_tables,
        diagnostics_df=diagnostics_df,
    )
    map_frame = city_da_gdf.merge(comp, on="da_code", how="inner").to_crs(4326)
    geojson = json.loads(map_frame[["da_code", "geometry"]].to_json())
    center_geom = map_frame.to_crs(3857).union_all().centroid
    center_geom = gpd.GeoSeries([center_geom], crs=3857).to_crs(4326).iloc[0]
    return px.choropleth_map(
        map_frame,
        geojson=geojson,
        locations="da_code",
        featureidkey="properties.da_code",
        color=value_col,
        hover_name="da_code",
        hover_data={
            "census_share": ":.2%",
            "synthetic_share": ":.2%",
            "share_diff": ":.2%",
        },
        center={"lat": center_geom.y, "lon": center_geom.x},
        zoom=12,
        opacity=0.65,
        color_continuous_scale="RdBu" if value_col == "share_diff" else "YlOrRd",
        title=title,
    )
