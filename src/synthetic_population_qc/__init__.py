"""Public package surface for the supported synthetic population workflow.

The package re-exports thin wrappers instead of importing every implementation
module eagerly. That keeps import-time dependencies light for notebooks and
small scripts that only need a subset of the workflow.
"""

from __future__ import annotations

from synthetic_population_qc.config import default_data_repo_root, default_geometry_dir, default_output_dir, project_root
from synthetic_population_qc.core import RawDataRoots, WorkflowSettings
from synthetic_population_qc.runs import (
    BasePopulationArtifacts,
    SyntheticPopulationRun,
    bundle_table_inventory,
    ensure_run_bundle,
    load_run_bundle,
)


def build_workflow_input_cache(*args, **kwargs):
    """Proxy to the processed-input cache builder."""
    from synthetic_population_qc.synth import build_workflow_input_cache as _impl

    return _impl(*args, **kwargs)


def build_base_population_cache(*args, **kwargs):
    """Proxy to the base-population cache builder."""
    from synthetic_population_qc.synth import build_base_population_cache as _impl

    return _impl(*args, **kwargs)


def run_energy_population_workflow(*args, **kwargs):
    """Proxy to the official bundle-first energy workflow runner."""
    from synthetic_population_qc.synth import run_energy_population_workflow as _impl

    return _impl(*args, **kwargs)


def build_workflow_plan_artifacts(*args, **kwargs):
    """Proxy to workflow planning artifact export."""
    from synthetic_population_qc.synth import build_workflow_plan_artifacts as _impl

    return _impl(*args, **kwargs)


def build_exploration_artifacts(*args, **kwargs):
    """Proxy to reusable exploration artifact generation."""
    from synthetic_population_qc.explore import build_exploration_artifacts as _impl

    return _impl(*args, **kwargs)


def export_exploration_plots(*args, **kwargs):
    """Proxy to HTML plot export for bundle exploration outputs."""
    from synthetic_population_qc.explore import export_exploration_plots as _impl

    return _impl(*args, **kwargs)


def export_map_outputs(*args, **kwargs):
    """Proxy to optional DA geometry map export."""
    from synthetic_population_qc.explore import export_map_outputs as _impl

    return _impl(*args, **kwargs)


def plot_metric_tvd(*args, **kwargs):
    """Proxy to the metric-TVD plotting helper."""
    from synthetic_population_qc.explore import plot_metric_tvd as _impl

    return _impl(*args, **kwargs)


def plot_support_assessment(*args, **kwargs):
    """Proxy to the support-assessment plotting helper."""
    from synthetic_population_qc.explore import plot_support_assessment as _impl

    return _impl(*args, **kwargs)


def plot_sparse_handling_report(*args, **kwargs):
    """Proxy to the sparse-handling plotting helper."""
    from synthetic_population_qc.explore import plot_sparse_handling_report as _impl

    return _impl(*args, **kwargs)


def plot_assignment_route_decisions(*args, **kwargs):
    """Proxy to the assignment-route plotting helper."""
    from synthetic_population_qc.explore import plot_assignment_route_decisions as _impl

    return _impl(*args, **kwargs)


def plot_household_coherence(*args, **kwargs):
    """Proxy to the household-coherence plotting helper."""
    from synthetic_population_qc.explore import plot_household_coherence as _impl

    return _impl(*args, **kwargs)


def plot_conditional_distribution(*args, **kwargs):
    """Proxy to the conditional-distribution plotting helper."""
    from synthetic_population_qc.explore import plot_conditional_distribution as _impl

    return _impl(*args, **kwargs)


def plot_commute_mode_by_age_labour(*args, **kwargs):
    """Proxy to the commute-mode exploration plotting helper."""
    from synthetic_population_qc.explore import plot_commute_mode_by_age_labour as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "RawDataRoots",
    "WorkflowSettings",
    "BasePopulationArtifacts",
    "SyntheticPopulationRun",
    "build_workflow_input_cache",
    "build_base_population_cache",
    "run_energy_population_workflow",
    "build_workflow_plan_artifacts",
    "load_run_bundle",
    "ensure_run_bundle",
    "bundle_table_inventory",
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
    "project_root",
    "default_data_repo_root",
    "default_geometry_dir",
    "default_output_dir",
]
