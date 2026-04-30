"""Reusable comparison, plotting, and map exports for run bundles."""

from synthetic_population_qc.explore.census_compare import (
    build_comparison_glossary,
    build_comparison_diagnostics,
    build_da_comparison_map,
    build_da_share_distribution,
    build_overall_comparison,
    plot_overall_comparison,
    plot_split_violin_comparison,
)
from synthetic_population_qc.explore.maps import export_map_outputs
from synthetic_population_qc.explore.plots import (
    export_exploration_plots,
    plot_commute_mode_by_age_labour,
    plot_conditional_distribution,
    plot_household_coherence,
    plot_metric_tvd,
    plot_assignment_route_decisions,
    plot_sparse_handling_report,
    plot_support_assessment,
)
from synthetic_population_qc.explore.workflow import build_exploration_artifacts

__all__ = [
    "build_comparison_diagnostics",
    "build_comparison_glossary",
    "build_overall_comparison",
    "build_da_share_distribution",
    "plot_overall_comparison",
    "plot_split_violin_comparison",
    "build_da_comparison_map",
    "build_exploration_artifacts",
    "export_exploration_plots",
    "export_map_outputs",
    "plot_metric_tvd",
    "plot_support_assessment",
    "plot_sparse_handling_report",
    "plot_assignment_route_decisions",
    "plot_household_coherence",
    "plot_conditional_distribution",
    "plot_commute_mode_by_age_labour",
]
