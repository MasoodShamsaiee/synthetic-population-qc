"""Validation and sparse-handling helpers used by the workflow."""

from synthetic_population_qc.sparse_handling import assign_attribute_with_fallback, conditional_support_weight
from synthetic_population_qc.support_assessment import build_support_assessment, support_strategy_map

__all__ = [
    "build_support_assessment",
    "support_strategy_map",
    "assign_attribute_with_fallback",
    "conditional_support_weight",
]
