"""Build planning artifacts that describe the workflow before synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synthetic_population_qc.core.types import RawDataRoots
from synthetic_population_qc.energy_workflow import _selected_da_codes
from synthetic_population_qc.joint_fit import build_joint_workflow_plan_artifacts
from synthetic_population_qc.mapping_audit import build_mapping_audit_df
from synthetic_population_qc.public_schema import public_attr_name, public_conditioning_cols
from synthetic_population_qc.runs.bundle import PlanningArtifacts, ProcessedArtifacts
from synthetic_population_qc.scope_selection import resolve_da_scope_codes
from synthetic_population_qc.support_assessment import build_support_assessment, support_strategy_map


def _load_context_tables(context_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all processed context tables from one standardized directory."""
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(context_dir.glob("*.parquet")):
        tables[path.stem] = pd.read_parquet(path)
    return tables


def build_workflow_plan_artifacts(
    *,
    raw_roots: RawDataRoots,
    processed: ProcessedArtifacts,
    output_dir: str | Path,
    method: str,
    province: str,
    geography_scope: str,
    max_das: int | None,
    da_codes: list[str] | None = None,
    da_scope_name: str | None = None,
    da_codes_file: str | Path | None = None,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> PlanningArtifacts:
    """Assemble the planning artifacts that drive the workflow."""
    plan_dir = Path(output_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)

    person_seed = pd.read_parquet(processed.person_seed_parquet)
    household_seed = pd.read_parquet(processed.household_seed_parquet)
    context_tables = _load_context_tables(processed.context_dir)
    explicit_da_codes = resolve_da_scope_codes(
        da_codes=da_codes,
        da_scope_name=da_scope_name,
        da_codes_file=da_codes_file,
    )
    # Planning and execution must agree on the DA list, so explicit scope
    # selection always wins over named geography convenience filters.
    da_codes = list(explicit_da_codes) if explicit_da_codes is not None else _selected_da_codes(
        geography_scope=geography_scope,
        data_root=raw_roots.data_root,
        context_tables=context_tables,
        max_das=max_das,
    )
    coverage_rows = []
    for table_name, table in context_tables.items():
        if "da_code" not in table.columns:
            continue
        table_codes = set(table["da_code"].astype(str).str.strip())
        coverage_rows.append(
            {
                "table_name": table_name,
                "n_unique_da_codes": int(len(table_codes)),
                "selected_da_overlap": int(len(table_codes & set(da_codes))),
            }
        )
    da_coverage_csv = plan_dir / "da_coverage.csv"
    coverage_df = pd.DataFrame(
        coverage_rows,
        columns=["table_name", "n_unique_da_codes", "selected_da_overlap"],
    )
    if not coverage_df.empty:
        coverage_df = coverage_df.sort_values("table_name")
    coverage_df.to_csv(da_coverage_csv, index=False)

    if method == "joint_ipu_v1":
        support_df, workflow_plan = build_joint_workflow_plan_artifacts(
            context_tables=context_tables,
            da_codes=da_codes,
        )
    else:
        support_df = build_support_assessment(
            person_seed_df=person_seed,
            household_seed_df=household_seed,
            context_tables=context_tables,
            da_codes=da_codes,
        )
        support_map = support_strategy_map(support_df)
        # The JSON plan is the contract downstream bundle consumers use to
        # understand which attributes were handled by each workflow route.
        workflow_plan = {
            "province": province,
            "geography_scope": geography_scope,
            "selected_da_count": int(len(da_codes)),
            "selected_da_codes": da_codes,
            "workflow_steps": [
                "prepare_inputs_and_scope",
                "assess_support_and_plan",
                "generate_base_population",
                "assign_household_attributes",
                "assign_person_attributes",
                "apply_sparse_fallbacks",
                "validate_and_explore",
            ],
            "core_generation_attributes": sorted(
                [
                    public_attr_name(attr)
                    for attr, cfg in support_map.items()
                    if cfg["assignment_route"] == "core_generation"
                ]
            ),
            "direct_household_attributes": sorted(
                [
                    public_attr_name(attr)
                    for attr, cfg in support_map.items()
                    if cfg["unit"] == "household" and cfg["assignment_route"] == "direct_household_assignment"
                ]
            ),
            "direct_person_attributes": sorted(
                [
                    public_attr_name(attr)
                    for attr, cfg in support_map.items()
                    if cfg["unit"] == "person" and cfg["assignment_route"] == "direct_person_assignment"
                ]
            ),
            "sparse_fallback_attributes": sorted(
                [public_attr_name(attr) for attr, cfg in support_map.items() if cfg["assignment_route"] == "sparse_fallback"]
            ),
            "all_workflow_attributes": [
                {
                    "attribute": public_attr_name(attr),
                    "unit": cfg["unit"],
                    "workflow_role": cfg["workflow_role"],
                    "assignment_route": cfg["assignment_route"],
                    "support_class": cfg["support_class"],
                }
                for attr, cfg in sorted(support_map.items())
            ],
            "support_diagnostics": {
                public_attr_name(attr): {
                    "unit": cfg["unit"],
                    "support_class": cfg["support_class"],
                    "recommended_cond_cols": public_conditioning_cols(cfg["recommended_cond_cols"]),
                    "fallback_ladder": [public_conditioning_cols(item) for item in cfg["fallback_ladder"]],
                }
                for attr, cfg in sorted(support_map.items())
            },
            "raw_inputs": {
                "data_root": str(raw_roots.data_root),
                "census_pumf_root": str(raw_roots.census_pumf_root),
                "housing_survey_root": str(raw_roots.housing_survey_root) if raw_roots.housing_survey_root else None,
                "geometry_root": str(raw_roots.geometry_root) if raw_roots.geometry_root else None,
                "base_population_path": str(raw_roots.base_population_path) if raw_roots.base_population_path else None,
            },
        }
    support_path = plan_dir / "support_classification.csv"
    support_report_df = support_df.copy()
    support_report_df["attribute"] = support_report_df["attribute"].map(public_attr_name)
    if "recommended_cond_cols_json" in support_report_df.columns:
        support_report_df["recommended_cond_cols_json"] = support_report_df["recommended_cond_cols_json"].apply(
            lambda value: json.dumps(public_conditioning_cols(json.loads(value)))
        )
    if "fallback_ladder_json" in support_report_df.columns:
        support_report_df["fallback_ladder_json"] = support_report_df["fallback_ladder_json"].apply(
            lambda value: json.dumps([public_conditioning_cols(item) for item in json.loads(value)])
        )
    support_report_df.to_csv(support_path, index=False)

    workflow_plan["workflow_method"] = method
    workflow_plan_json = plan_dir / "workflow_plan.json"
    workflow_plan_json.write_text(json.dumps(workflow_plan, indent=2), encoding="utf-8")

    mapping_audit_csv = plan_dir / "parameter_mapping_audit.csv"
    build_mapping_audit_df(
        age_group_scheme=age_group_scheme,
        age_group_breaks=age_group_breaks,
    ).to_csv(mapping_audit_csv, index=False)

    return PlanningArtifacts(
        workflow_plan_json=workflow_plan_json,
        da_coverage_csv=da_coverage_csv,
        support_classification_csv=support_path,
        mapping_audit_csv=mapping_audit_csv,
    )
