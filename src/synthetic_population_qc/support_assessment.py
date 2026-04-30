"""Support diagnostics for deciding how each workflow attribute should be assigned.

The workflow separates attributes into:
- core-generation attributes that come from the base population
- direct-assignment attributes that can be synthesized from DA marginals
- sparse-fallback attributes that should use fallback conditioning
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from synthetic_population_qc.enrichment import HOUSEHOLD_ATTR_LABELS, PERSON_ATTR_LABELS


@dataclass(frozen=True)
class SupportSpec:
    """Static recipe describing how one workflow attribute should be evaluated."""

    attribute: str
    unit: str
    table_name: str
    seed_source: str
    seed_column: str
    restricted_universe: bool
    cond_cols: tuple[str, ...]
    fallback_ladder: tuple[tuple[str, ...], ...]
    workflow_role: str = "extension"
    requires_da_marginals: bool = True


SUPPORT_SPECS: tuple[SupportSpec, ...] = (
    SupportSpec(
        attribute="sex",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="sex",
        restricted_universe=False,
        cond_cols=tuple(),
        fallback_ladder=(tuple(),),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="age_group",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="age_group",
        restricted_universe=False,
        cond_cols=("sex",),
        fallback_ladder=(("sex",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="education_level",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="education_level",
        restricted_universe=False,
        cond_cols=("sex", "age_group"),
        fallback_ladder=(("sex", "age_group"), ("sex",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="labour_force_status",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="labour_force_status",
        restricted_universe=False,
        cond_cols=("sex", "age_group"),
        fallback_ladder=(("sex", "age_group"), ("sex",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="household_income",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="household_income",
        restricted_universe=False,
        cond_cols=("sex", "age_group", "labour_force_status"),
        fallback_ladder=(("sex", "age_group", "labour_force_status"), ("sex", "age_group"), ("sex",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="family_status",
        unit="person",
        table_name="core_population",
        seed_source="person_seed",
        seed_column="family_status",
        restricted_universe=False,
        cond_cols=("sex", "age_group"),
        fallback_ladder=(("sex", "age_group"), ("sex",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="household_size",
        unit="household",
        table_name="core_population",
        seed_source="household_seed",
        seed_column="household_size",
        restricted_universe=False,
        cond_cols=tuple(),
        fallback_ladder=(tuple(),),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="household_type",
        unit="household",
        table_name="core_population",
        seed_source="household_seed",
        seed_column="household_type",
        restricted_universe=False,
        cond_cols=("household_size",),
        fallback_ladder=(("household_size",), tuple()),
        workflow_role="core_generation",
        requires_da_marginals=False,
    ),
    SupportSpec(
        attribute="citizenship_status",
        unit="person",
        table_name="immigration_citizenship",
        seed_source="person_seed",
        seed_column="citizenship_status",
        restricted_universe=False,
        cond_cols=("sex", "age_group", "labour_force_status"),
        fallback_ladder=(("sex", "age_group", "labour_force_status"), ("sex", "age_group"), ("sex",), tuple()),
    ),
    SupportSpec(
        attribute="immigrant_status",
        unit="person",
        table_name="immigration_citizenship",
        seed_source="person_seed",
        seed_column="immigrant_status",
        restricted_universe=False,
        cond_cols=("sex", "age_group", "labour_force_status"),
        fallback_ladder=(("sex", "age_group", "labour_force_status"), ("sex", "age_group"), ("sex",), tuple()),
    ),
    SupportSpec(
        attribute="commute_mode",
        unit="person",
        table_name="commute",
        seed_source="person_seed",
        seed_column="commute_mode_group",
        restricted_universe=True,
        cond_cols=("sex", "age_group"),
        fallback_ladder=(("sex", "age_group"), ("sex",), tuple()),
    ),
    SupportSpec(
        attribute="commute_duration",
        unit="person",
        table_name="commute",
        seed_source="person_seed",
        seed_column="commute_duration",
        restricted_universe=True,
        cond_cols=("sex", "age_group"),
        fallback_ladder=(("sex", "age_group"), ("sex",), tuple()),
    ),
    SupportSpec(
        attribute="dwelling_type",
        unit="household",
        table_name="dwelling_characteristics",
        seed_source="household_seed",
        seed_column="dwelling_type",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
    SupportSpec(
        attribute="tenure",
        unit="household",
        table_name="housing",
        seed_source="household_seed",
        seed_column="tenure",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
    SupportSpec(
        attribute="bedrooms",
        unit="household",
        table_name="housing",
        seed_source="household_seed",
        seed_column="bedrooms",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
    SupportSpec(
        attribute="period_built",
        unit="household",
        table_name="housing",
        seed_source="household_seed",
        seed_column="period_built",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
    SupportSpec(
        attribute="dwelling_condition",
        unit="household",
        table_name="housing",
        seed_source="household_seed",
        seed_column="dwelling_condition",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
    SupportSpec(
        attribute="core_housing_need",
        unit="household",
        table_name="housing",
        seed_source="household_seed",
        seed_column="core_housing_need",
        restricted_universe=False,
        cond_cols=("household_size", "household_type"),
        fallback_ladder=(("household_size", "household_type"), ("household_size",), tuple()),
    ),
)


def _numeric_total(df: pd.DataFrame, columns: list[str]) -> float:
    """Sum numeric marginal columns while tolerating missing tables/values."""
    if df.empty:
        return 0.0
    total = 0.0
    for col in columns:
        if col in df.columns:
            total += float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())
    return total


def _min_positive_weight(seed_df: pd.DataFrame, cols: list[str], weight_col: str) -> float:
    """Return the weakest non-zero conditional cell in the weighted seed."""
    if seed_df.empty:
        return 0.0
    grouped = (
        seed_df[cols + [weight_col]]
        .dropna(subset=cols)
        .groupby(cols, dropna=False)[weight_col]
        .sum()
        .astype(float)
    )
    grouped = grouped[grouped > 0]
    return float(grouped.min()) if not grouped.empty else 0.0


def _support_class(
    *,
    da_marginal_available: bool,
    seed_available: bool,
    n_seed_categories: int,
    min_category_weight: float,
    min_conditional_weight: float,
    restricted_universe: bool,
) -> str:
    """Collapse support diagnostics into the workflow's stable/sparse labels."""
    if not da_marginal_available or not seed_available or n_seed_categories <= 1:
        return "highly_sparse"
    if min_category_weight >= 100.0 and min_conditional_weight >= 25.0 and not restricted_universe:
        return "stable"
    if min_category_weight >= 30.0 and min_conditional_weight >= 10.0:
        return "moderately_sparse" if restricted_universe else "stable"
    if min_category_weight >= 10.0 and min_conditional_weight >= 3.0:
        return "moderately_sparse"
    return "highly_sparse"


def _assignment_route(unit: str, support_class: str, restricted_universe: bool, workflow_role: str) -> str:
    """Map support strength onto the route that should assign the attribute."""
    if workflow_role == "core_generation":
        return "core_generation"
    if support_class == "stable" and not restricted_universe:
        return "direct_person_assignment" if unit == "person" else "direct_household_assignment"
    return "sparse_fallback"


def build_support_assessment(
    *,
    person_seed_df: pd.DataFrame,
    household_seed_df: pd.DataFrame,
    context_tables: dict[str, pd.DataFrame],
    da_codes: list[str] | None = None,
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Classify each workflow attribute by support strength and assignment route."""
    rows: list[dict[str, object]] = []
    da_filter = set(map(str, da_codes or []))
    for spec in SUPPORT_SPECS:
        seed_df = person_seed_df if spec.seed_source == "person_seed" else household_seed_df
        work = seed_df.copy()
        if spec.seed_column not in work.columns:
            seed_available = False
            n_seed_categories = 0
            min_category_weight = 0.0
            min_conditional_weight = 0.0
        else:
            seed_available = bool(work[spec.seed_column].notna().any())
            # We care about weighted support, not raw row counts, because the
            # seed microdata already carries survey expansion weights.
            value_counts = (
                work[[spec.seed_column, weight_col]]
                .dropna(subset=[spec.seed_column])
                .groupby(spec.seed_column, dropna=False)[weight_col]
                .sum()
                .astype(float)
            )
            n_seed_categories = int(len(value_counts))
            min_category_weight = float(value_counts.min()) if not value_counts.empty else 0.0
            min_conditional_weight = _min_positive_weight(
                work,
                list(spec.cond_cols) + [spec.seed_column],
                weight_col,
            )

        table = context_tables.get(spec.table_name, pd.DataFrame()).copy()
        if da_filter and "da_code" in table.columns:
            table = table.loc[table["da_code"].astype(str).isin(da_filter)].copy()
        label_map = PERSON_ATTR_LABELS.get(spec.attribute) if spec.unit == "person" else HOUSEHOLD_ATTR_LABELS.get(spec.attribute)
        marginal_columns = sorted(set((label_map or {}).values()))
        # Core-generation attributes intentionally skip DA-marginal checks
        # because they are guaranteed by the base-population stage.
        da_total = _numeric_total(table, marginal_columns) if spec.requires_da_marginals else pd.NA
        da_marginal_available = bool(da_total > 0) if spec.requires_da_marginals else pd.NA
        support_class = _support_class(
            da_marginal_available=bool(da_marginal_available) if spec.requires_da_marginals else True,
            seed_available=seed_available,
            n_seed_categories=n_seed_categories,
            min_category_weight=min_category_weight,
            min_conditional_weight=min_conditional_weight,
            restricted_universe=spec.restricted_universe,
        )
        rows.append(
            {
                "attribute": spec.attribute,
                "unit": spec.unit,
                "seed_source": spec.seed_source,
                "seed_column": spec.seed_column,
                "table_name": spec.table_name,
                "da_marginal_available": da_marginal_available,
                "seed_available": seed_available,
                "da_total": da_total,
                "n_seed_categories": n_seed_categories,
                "min_category_weight": min_category_weight,
                "min_conditional_weight": min_conditional_weight,
                "restricted_universe": spec.restricted_universe,
                "support_class": support_class,
                "assignment_route": _assignment_route(spec.unit, support_class, spec.restricted_universe, spec.workflow_role),
                "recommended_cond_cols_json": json.dumps(list(spec.cond_cols)),
                "fallback_ladder_json": json.dumps([list(x) for x in spec.fallback_ladder]),
                "workflow_role": spec.workflow_role,
                "requires_da_marginals": spec.requires_da_marginals,
            }
        )
    return pd.DataFrame(rows)


def support_strategy_map(support_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Convert the support table into a dict keyed by workflow attribute name."""
    if support_df.empty:
        return {}
    strategies: dict[str, dict[str, object]] = {}
    for row in support_df.to_dict(orient="records"):
        strategies[str(row["attribute"])] = {
            **row,
            "recommended_cond_cols": json.loads(row.get("recommended_cond_cols_json", "[]")),
            "fallback_ladder": json.loads(row.get("fallback_ladder_json", "[]")),
        }
    return strategies
