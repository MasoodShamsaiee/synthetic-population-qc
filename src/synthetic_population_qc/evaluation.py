from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from synthetic_population_qc.workflow import norm_code


def _safe_num(x):
    try:
        v = float(x)
        return np.nan if pd.isna(v) else v
    except Exception:
        return np.nan


def _safe_div(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def _sum_varids(census_da: pd.DataFrame, varids: list[int]) -> float:
    vals = []
    for vid in varids:
        s = census_da.loc[census_da["variableId"] == vid, "total"]
        vals.append(np.nan if s.empty else _safe_num(s.iloc[0]))
    if all(pd.isna(v) for v in vals):
        return np.nan
    return float(np.nansum(vals))


def build_da_core_metrics(
    *,
    syn_with_hh: pd.DataFrame,
    census: pd.DataFrame,
    total_vb_id: int,
    total_age_by_sex_vb_id: int,
    total_hh_vb_id: int,
    area_col: str = "area",
) -> pd.DataFrame:
    syn = syn_with_hh.copy()
    syn[area_col] = syn[area_col].map(norm_code)
    c = census.copy()
    c["geocode_norm"] = c["geocode"].map(norm_code)

    rows: list[dict] = []
    for da, g in syn.groupby(area_col, sort=True):
        if da is None:
            continue
        c_da = c.loc[c["geocode_norm"] == da]
        if c_da.empty:
            continue

        pop_syn = float(len(g))
        hh_syn = float(g.loc[g["HID"] != -1, "HID"].nunique()) if "HID" in g.columns else np.nan

        pop_cen = _sum_varids(c_da, [total_vb_id])
        hh_cen = _sum_varids(c_da, [total_hh_vb_id])
        pop_age_cen = _sum_varids(c_da, [total_age_by_sex_vb_id])

        rows.append(
            {
                "da_code": da,
                "pop_syn": pop_syn,
                "pop_cen": pop_cen,
                "pop_age_cen": pop_age_cen,
                "hh_syn": hh_syn,
                "hh_cen": hh_cen,
                "pop_abs_err": abs(pop_syn - pop_cen) if not pd.isna(pop_cen) else np.nan,
                "hh_abs_err": abs(hh_syn - hh_cen) if not pd.isna(hh_cen) else np.nan,
                "pop_pct_err": 100.0 * _safe_div(abs(pop_syn - pop_cen), pop_cen),
                "hh_pct_err": 100.0 * _safe_div(abs(hh_syn - hh_cen), hh_cen),
            }
        )

    return pd.DataFrame(rows).sort_values("da_code").reset_index(drop=True)


def summarize_errors(metrics_df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in ["pop_pct_err", "hh_pct_err", "pop_abs_err", "hh_abs_err"]:
        if col in metrics_df.columns and not metrics_df[col].dropna().empty:
            s = metrics_df[col].dropna()
            out[col] = {
                "mean": float(s.mean()),
                "median": float(s.median()),
                "p90": float(s.quantile(0.9)),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
            }
    return pd.DataFrame(out).T


def plot_error_hist(metrics_df: pd.DataFrame, col: str = "hh_pct_err", nbins: int = 40):
    fig = px.histogram(metrics_df, x=col, nbins=nbins, title=f"Distribution of {col}")
    fig.update_layout(template="plotly_white")
    return fig


def plot_top_da_errors(metrics_df: pd.DataFrame, col: str = "hh_pct_err", n: int = 30):
    d = metrics_df.sort_values(col, ascending=False).head(n).copy()
    fig = px.bar(d, x="da_code", y=col, title=f"Top {n} DAs by {col}")
    fig.update_layout(template="plotly_white")
    fig.update_xaxes(tickangle=45)
    return fig


def plot_syn_vs_census_scatter(metrics_df: pd.DataFrame, which: str = "pop"):
    if which == "pop":
        x, y = "pop_cen", "pop_syn"
    else:
        x, y = "hh_cen", "hh_syn"
    d = metrics_df[[x, y, "da_code"]].dropna()
    fig = px.scatter(d, x=x, y=y, hover_data=["da_code"], title=f"{which.upper()}: synthetic vs census")
    if not d.empty:
        maxv = float(np.nanmax(np.r_[d[x].values, d[y].values]))
        fig.add_trace(go.Scatter(x=[0, maxv], y=[0, maxv], mode="lines", name="y=x"))
    fig.update_layout(template="plotly_white")
    return fig


def plot_metric_box(metrics_df: pd.DataFrame, cols=("pop_pct_err", "hh_pct_err")):
    long = metrics_df[list(cols)].melt(var_name="metric", value_name="value")
    fig = px.box(long, x="metric", y="value", points="outliers", title="Error boxplots")
    fig.update_layout(template="plotly_white")
    return fig


def plot_error_ecdf(metrics_df: pd.DataFrame, col: str = "hh_pct_err"):
    d = metrics_df[[col]].dropna().sort_values(col).reset_index(drop=True)
    if d.empty:
        return go.Figure()
    d["ecdf"] = (np.arange(len(d)) + 1) / len(d)
    fig = px.line(d, x=col, y="ecdf", title=f"ECDF of {col}")
    fig.update_layout(template="plotly_white")
    return fig


def run_eval_suite(
    *,
    syn_with_hh: pd.DataFrame,
    census: pd.DataFrame,
    total_vb_id: int,
    total_age_by_sex_vb_id: int,
    total_hh_vb_id: int,
    area_col: str = "area",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics = build_da_core_metrics(
        syn_with_hh=syn_with_hh,
        census=census,
        total_vb_id=total_vb_id,
        total_age_by_sex_vb_id=total_age_by_sex_vb_id,
        total_hh_vb_id=total_hh_vb_id,
        area_col=area_col,
    )
    summary = summarize_errors(metrics)
    figs = {
        "hist_hh_pct_err": plot_error_hist(metrics, "hh_pct_err"),
        "hist_pop_pct_err": plot_error_hist(metrics, "pop_pct_err"),
        "top_hh_pct_err": plot_top_da_errors(metrics, "hh_pct_err", n=30),
        "top_pop_pct_err": plot_top_da_errors(metrics, "pop_pct_err", n=30),
        "scatter_pop": plot_syn_vs_census_scatter(metrics, "pop"),
        "scatter_hh": plot_syn_vs_census_scatter(metrics, "hh"),
        "box_errors": plot_metric_box(metrics),
        "ecdf_hh_pct_err": plot_error_ecdf(metrics, "hh_pct_err"),
        "ecdf_pop_pct_err": plot_error_ecdf(metrics, "pop_pct_err"),
    }
    return metrics, summary, figs


def detect_geo_da_col(gdf, preferred=None) -> str:
    preferred = preferred or [
        "DAUID",
        "DAUID_ADIDU",
        "DAUID_2021",
        "dauid",
        "ADAUID_ADAIDU",
        "ADAUID",
        "GEOUID",
        "geocode",
        "da_code",
    ]
    for c in preferred:
        if c in gdf.columns:
            return c
    raise KeyError(f"No DA column detected. Columns={list(gdf.columns)}")


def build_da_gdf(
    *,
    project_root: str | Path | None = None,
    da_shp: str | Path | None = None,
) -> tuple:
    import geopandas as gpd

    if da_shp is None:
        root = Path(project_root) if project_root is not None else _project_root()
        da_shp = root / "data" / "raw" / "geometry" / "lda_000b21a_e" / "lda_000b21a_e.shp"
    da_shp = Path(da_shp)
    if not da_shp.exists():
        raise FileNotFoundError(f"DA shapefile not found: {da_shp}")

    gdf = gpd.read_file(da_shp)
    da_col = detect_geo_da_col(gdf)
    gdf["da_code"] = gdf[da_col].map(norm_code)
    gdf = gdf.loc[gdf["da_code"].notna()].copy()
    if gdf.crs is not None and str(gdf.crs).lower() not in {"epsg:4326", "4326"}:
        gdf = gdf.to_crs(epsg=4326)
    return gdf, da_col


def map_metric_polygons(
    *,
    metrics_df: pd.DataFrame,
    gdf,
    geo_da_col: str = "da_code",
    metric_col: str = "hh_pct_err",
    zoom: float = 10.2,
):
    g = gdf.copy()
    if geo_da_col != "da_code":
        g["da_code"] = g[geo_da_col].map(norm_code)
    else:
        g["da_code"] = g["da_code"].map(norm_code)

    m = metrics_df.copy()
    m["da_code"] = m["da_code"].map(norm_code)
    g = g.merge(m, on="da_code", how="left").reset_index(drop=True)
    g["plot_id"] = g.index.astype(str)

    geojson = g.__geo_interface__
    center = {"lat": float(g.geometry.centroid.y.mean()), "lon": float(g.geometry.centroid.x.mean())}

    fig = px.choropleth_mapbox(
        g,
        geojson=geojson,
        locations="plot_id",
        featureidkey="properties.plot_id",
        color=metric_col,
        hover_name="da_code",
        hover_data=["pop_syn", "pop_cen", "hh_syn", "hh_cen"],
        center=center,
        zoom=zoom,
        mapbox_style="carto-positron",
        height=750,
        title=f"DA polygon map: {metric_col}",
        color_continuous_scale="Turbo",
    )
    return fig


def map_all_key_metrics_polygons(*, metrics_df: pd.DataFrame, gdf, geo_da_col: str = "da_code", zoom: float = 10.2) -> dict:
    return {
        "map_pop_pct_err": map_metric_polygons(metrics_df=metrics_df, gdf=gdf, geo_da_col=geo_da_col, metric_col="pop_pct_err", zoom=zoom),
        "map_hh_pct_err": map_metric_polygons(metrics_df=metrics_df, gdf=gdf, geo_da_col=geo_da_col, metric_col="hh_pct_err", zoom=zoom),
        "map_pop_abs_err": map_metric_polygons(metrics_df=metrics_df, gdf=gdf, geo_da_col=geo_da_col, metric_col="pop_abs_err", zoom=zoom),
        "map_hh_abs_err": map_metric_polygons(metrics_df=metrics_df, gdf=gdf, geo_da_col=geo_da_col, metric_col="hh_abs_err", zoom=zoom),
    }


def map_metric_points_from_lookup(
    *,
    metrics_df: pd.DataFrame,
    lookup_csv_path: str | Path,
    da_col_lookup: str = "DAUID_ADIDU",
    lat_col: str = "DARPLAT_ADLAT",
    lon_col: str = "DARPLONG_ADLONG",
    metric_col: str = "hh_pct_err",
    zoom: float = 10.0,
):
    lookup = pd.read_csv(Path(lookup_csv_path), encoding="ISO-8859-1", low_memory=False)
    lookup["da_code"] = lookup[da_col_lookup].map(norm_code)
    lookup[lat_col] = pd.to_numeric(lookup[lat_col], errors="coerce")
    lookup[lon_col] = pd.to_numeric(lookup[lon_col], errors="coerce")

    m = metrics_df.copy()
    m["da_code"] = m["da_code"].map(norm_code)
    d = m.merge(lookup[["da_code", lat_col, lon_col]], on="da_code", how="left").dropna(subset=[lat_col, lon_col])

    fig = px.scatter_mapbox(
        d,
        lat=lat_col,
        lon=lon_col,
        color=metric_col,
        hover_data=["da_code", "pop_syn", "pop_cen", "hh_syn", "hh_cen"],
        zoom=zoom,
        height=700,
        title=f"DA point map: {metric_col}",
        color_continuous_scale="Turbo",
    )
    fig.update_layout(mapbox_style="carto-positron")
    return fig


def _project_root(cwd: Path | None = None) -> Path:
    here = (cwd or Path.cwd()).resolve()
    return here.parent if here.name.lower() == "notebooks" else here
