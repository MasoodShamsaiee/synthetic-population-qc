"""Optional DA geometry map exports for exploration bundles."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px


def _load_geometry(geography_scope: str, geometry_root: str | Path | None) -> gpd.GeoDataFrame | None:
    """Load DA geometries for the named geography scope when available."""
    try:
        from urban_energy_core import load_city_da_geojsons
    except Exception:
        return None
    try:
        geometry_lookup = load_city_da_geojsons(geometry_dir=geometry_root, show_progress=False)
        gdf = geometry_lookup.get(geography_scope)
    except Exception:
        return None
    if gdf is None or gdf.empty:
        return None
    out = gdf.copy()
    if "DAUID" in out.columns and "da_code" not in out.columns:
        out["da_code"] = out["DAUID"].astype(str).str.strip()
    return out


def _write_map(fig, path: Path) -> Path:
    """Persist one map figure as an HTML artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def export_map_outputs(
    *,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    geography_scope: str,
    geometry_root: str | Path | None,
    output_dir: str | Path,
) -> dict[str, Path | None]:
    """Write DA-level exploration maps for the selected geography scope."""
    out_dir = Path(output_dir)
    gdf = _load_geometry(geography_scope=geography_scope, geometry_root=geometry_root)
    if gdf is None:
        return {"households_map_html": None, "owners_share_map_html": None}

    selected_da_codes = set(households_df.get("area", pd.Series(dtype="object")).dropna().astype(str))
    if selected_da_codes and "da_code" in gdf.columns:
        gdf = gdf.loc[gdf["da_code"].astype(str).isin(selected_da_codes)].copy()
    if gdf.empty:
        return {"households_map_html": None, "owners_share_map_html": None}

    household_counts = households_df.groupby("area", dropna=False).size().rename("household_count").reset_index()
    owner_share = (
        households_df.assign(is_owner=(households_df.get("tenure") == "owner").astype(float))
        .groupby("area", dropna=False)["is_owner"]
        .mean()
        .rename("owner_share")
        .reset_index()
    )
    merged = gdf.merge(household_counts, left_on="da_code", right_on="area", how="left").merge(
        owner_share, left_on="da_code", right_on="area", how="left", suffixes=("", "_owner")
    )
    geojson = merged.set_index("da_code").geometry.__geo_interface__

    households_fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="da_code",
        color="household_count",
        hover_name="da_code",
        title="Synthetic Household Count by DA",
    )
    households_fig.update_geos(fitbounds="locations", visible=False)

    owners_fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="da_code",
        color="owner_share",
        hover_name="da_code",
        title="Owner Share by DA",
    )
    owners_fig.update_geos(fitbounds="locations", visible=False)

    return {
        "households_map_html": _write_map(households_fig, out_dir / "households_by_da_map.html"),
        "owners_share_map_html": _write_map(owners_fig, out_dir / "owner_share_by_da_map.html"),
    }
