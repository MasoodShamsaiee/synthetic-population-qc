"""Plot builders for standardized exploration outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _write_figure(fig: go.Figure, path: Path) -> Path:
    """Persist one Plotly figure as an HTML artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def plot_metric_tvd(summary_df: pd.DataFrame) -> go.Figure:
    """Plot attribute-level TVD metrics from the validation summary."""
    frame = summary_df.copy()
    if frame.empty:
        return px.bar(title="No metric summary available")
    frame = frame.sort_values(["unit", "tvd", "attribute"], ascending=[True, False, True])
    return px.bar(
        frame,
        x="attribute",
        y="tvd",
        color="unit",
        barmode="group",
        title="Attribute-Level TVD",
        hover_data=["mae_pp", "max_abs_pp"],
    )


def plot_support_assessment(support_df: pd.DataFrame) -> go.Figure:
    """Plot minimum conditional support weights by attribute."""
    frame = support_df.copy()
    if frame.empty:
        return px.bar(title="No support assessment available")
    frame["label"] = frame["attribute"] + " (" + frame["unit"] + ")"
    return px.bar(
        frame.sort_values(["unit", "min_conditional_weight"], ascending=[True, False]),
        x="label",
        y="min_conditional_weight",
        color="support_class",
        title="Support Assessment Minimum Conditional Weight",
        hover_data=["min_category_weight", "assignment_route"],
    )


def plot_sparse_handling_report(sparse_df: pd.DataFrame) -> go.Figure:
    """Plot sparse-handling fallback decisions by attribute."""
    frame = sparse_df.copy()
    if frame.empty:
        return px.bar(title="No sparse-handling report available")
    return px.histogram(
        frame,
        x="attribute",
        color="fallback_rank",
        title="Sparse Handling Decisions",
        hover_data=["area", "unit", "used_global_fallback"],
    )


def plot_assignment_route_decisions(route_df: pd.DataFrame) -> go.Figure:
    """Plot assignment-route selections recorded during synthesis."""
    frame = route_df.copy()
    if frame.empty:
        return px.bar(title="No assignment-route decisions available")
    return px.histogram(
        frame,
        x="attribute",
        color="selected_route",
        title="Assignment Route Decisions",
        hover_data=["area", "unit", "planned_route", "downgraded_to_sparse"],
    )


def plot_household_coherence(audit_df: pd.DataFrame) -> go.Figure:
    """Plot household-coherence issues recorded during validation."""
    frame = audit_df.copy()
    if frame.empty:
        return px.bar(title="Household coherence audit passed")
    return px.histogram(
        frame,
        x="coherence_issue",
        color="coherence_issue",
        title="Household Coherence Audit",
        hover_data=["area", "household_id"],
    )


def plot_conditional_distribution(
    df: pd.DataFrame,
    *,
    given: str,
    value: str,
    title: str,
) -> go.Figure:
    """Plot the conditional share distribution for one pair of attributes."""
    if df.empty or given not in df.columns or value not in df.columns:
        return px.bar(title=title)
    frame = df[[given, value]].dropna().copy()
    if frame.empty:
        return px.bar(title=title)
    counts = frame.value_counts().rename("n").reset_index()
    totals = counts.groupby(given)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return px.bar(
        counts,
        x=given,
        y="share",
        color=value,
        barmode="group",
        title=title,
        hover_data=["n"],
    )


def plot_commute_mode_by_age_labour(people_df: pd.DataFrame) -> go.Figure:
    """Plot commute-mode shares by labour-force status and age group."""
    if people_df.empty:
        return px.bar(title="Commute mode by labour-force status and age group")
    frame = people_df.copy()
    if "labour_force_status" not in frame.columns or "age_group" not in frame.columns or "commute_mode" not in frame.columns:
        return px.bar(title="Commute mode by labour-force status and age group")
    frame = frame[["labour_force_status", "age_group", "commute_mode"]].dropna()
    if frame.empty:
        return px.bar(title="Commute mode by labour-force status and age group")
    grouped = frame.value_counts().rename("n").reset_index()
    grouped["labour_age"] = grouped["labour_force_status"].astype(str) + " / " + grouped["age_group"].astype(str)
    grouped["share"] = grouped["n"] / grouped.groupby("labour_age")["n"].transform("sum")
    return px.bar(
        grouped,
        x="labour_age",
        y="share",
        color="commute_mode",
        title="Commute Mode by Labour-Force Status and Age Group",
        hover_data=["n"],
    )


def export_exploration_plots(
    *,
    summary_df: pd.DataFrame,
    support_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    route_df: pd.DataFrame,
    coherence_df: pd.DataFrame,
    people_df: pd.DataFrame,
    households_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the standard HTML exploration plots for one workflow bundle."""
    out_dir = Path(output_dir)
    figures = {
        "metric_plot_html": _write_figure(plot_metric_tvd(summary_df), out_dir / "attribute_tvd.html"),
        "support_plot_html": _write_figure(plot_support_assessment(support_df), out_dir / "support_assessment.html"),
        "sparse_plot_html": _write_figure(plot_sparse_handling_report(sparse_df), out_dir / "sparse_handling.html"),
        "assignment_route_plot_html": _write_figure(
            plot_assignment_route_decisions(route_df),
            out_dir / "assignment_route_decisions.html",
        ),
        "coherence_plot_html": _write_figure(plot_household_coherence(coherence_df), out_dir / "household_coherence.html"),
        "dwelling_type_by_household_size_html": _write_figure(
            plot_conditional_distribution(
                households_df,
                given="household_size",
                value="dwelling_type",
                title="Dwelling Type by Household Size",
            ),
            out_dir / "dwelling_type_by_household_size.html",
        ),
        "tenure_by_household_type_html": _write_figure(
            plot_conditional_distribution(
                households_df,
                given="household_type",
                value="tenure",
                title="Tenure by Household Type",
            ),
            out_dir / "tenure_by_household_type.html",
        ),
        "period_built_by_dwelling_type_html": _write_figure(
            plot_conditional_distribution(
                households_df,
                given="dwelling_type",
                value="period_built",
                title="Period Built by Dwelling Type",
            ),
            out_dir / "period_built_by_dwelling_type.html",
        ),
        "commute_mode_by_age_labour_html": _write_figure(
            plot_commute_mode_by_age_labour(people_df),
            out_dir / "commute_mode_by_age_labour.html",
        ),
    }
    return figures
