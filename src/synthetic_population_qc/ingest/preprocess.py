"""Build harmonized processed-input caches for the workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from synthetic_population_qc.context_tables import load_context_tables
from synthetic_population_qc.core.types import RawDataRoots
from synthetic_population_qc.seed_preparation import PreparedSeedArtifacts, export_prepared_seed_artifacts
from synthetic_population_qc.runs.bundle import ProcessedArtifacts, load_processed_artifacts


def _normalize_da_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize `da_code` formatting before writing processed context tables."""
    out = df.copy()
    if "da_code" in out.columns:
        out["da_code"] = out["da_code"].astype(str).str.strip().str.replace(".0", "", regex=False)
    return out


def export_processed_inputs(
    raw_roots: RawDataRoots,
    *,
    output_dir: str | Path,
    province: str = "24",
) -> ProcessedArtifacts:
    """Export seed artifacts and context tables into the standard processed bundle."""
    processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    seed_dir = processed_dir / "seeds"
    context_dir = processed_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)

    prepared_artifacts: PreparedSeedArtifacts = export_prepared_seed_artifacts(
        data_root=raw_roots.data_root,
        census_pumf_root=raw_roots.census_pumf_root,
        housing_survey_root=raw_roots.housing_survey_root,
        output_dir=seed_dir,
        province=province,
    )

    context_tables = load_context_tables(raw_roots.data_root)
    context_rows: list[dict[str, object]] = []
    coverage_codes: set[str] = set()
    for table_name, table in context_tables.items():
        normalized = _normalize_da_codes(table)
        if "da_code" in normalized.columns:
            coverage_codes.update(x for x in normalized["da_code"].dropna().astype(str) if x.strip())
        parquet_path = context_dir / f"{table_name}.parquet"
        csv_path = context_dir / f"{table_name}.csv"
        normalized.to_parquet(parquet_path, index=False)
        normalized.to_csv(csv_path, index=False)
        context_rows.append(
            {
                "table_name": table_name,
                "rows": int(len(normalized)),
                "columns": int(len(normalized.columns)),
                "has_da_code": bool("da_code" in normalized.columns),
                "csv_path": str(csv_path),
                "parquet_path": str(parquet_path),
            }
        )

    context_manifest_csv = processed_dir / "context_manifest.csv"
    pd.DataFrame(context_rows).sort_values("table_name").to_csv(context_manifest_csv, index=False)

    preprocessing_audit_csv = processed_dir / "preprocessing_audit.csv"
    audit_rows = [
        {
            "artifact": "person_seed",
            "path": str(prepared_artifacts.person_seed_parquet),
            "exists": prepared_artifacts.person_seed_parquet.exists(),
        },
        {
            "artifact": "hierarchical_household_seed",
            "path": str(prepared_artifacts.hierarchical_household_seed_parquet),
            "exists": prepared_artifacts.hierarchical_household_seed_parquet.exists(),
        },
        {
            "artifact": "chs_household_seed",
            "path": str(prepared_artifacts.chs_household_seed_parquet) if prepared_artifacts.chs_household_seed_parquet else "",
            "exists": bool(prepared_artifacts.chs_household_seed_parquet and prepared_artifacts.chs_household_seed_parquet.exists()),
        },
        {
            "artifact": "harmonized_context_tables",
            "path": str(context_dir),
            "exists": context_dir.exists(),
        },
        {
            "artifact": "unique_da_codes_preview",
            "path": json.dumps(sorted(coverage_codes)[:25]),
            "exists": bool(coverage_codes),
        },
    ]
    pd.DataFrame(audit_rows).to_csv(preprocessing_audit_csv, index=False)

    return ProcessedArtifacts(
        input_contract_csv=prepared_artifacts.input_contract_csv,
        preprocessing_audit_csv=preprocessing_audit_csv,
        person_seed_parquet=prepared_artifacts.person_seed_parquet,
        household_seed_parquet=prepared_artifacts.hierarchical_household_seed_parquet,
        chs_household_seed_parquet=prepared_artifacts.chs_household_seed_parquet,
        person_seed_summary_csv=prepared_artifacts.person_summary_csv,
        household_seed_summary_csv=prepared_artifacts.household_summary_csv,
        context_dir=context_dir,
        context_manifest_csv=context_manifest_csv,
    )


def load_processed_context_tables(processed: ProcessedArtifacts | str | Path) -> dict[str, pd.DataFrame]:
    """Load processed context tables back from a standardized processed bundle."""
    artifacts = (
        processed
        if isinstance(processed, ProcessedArtifacts)
        else load_processed_artifacts(Path(processed))
    )
    tables: dict[str, pd.DataFrame] = {}
    manifest = pd.read_csv(artifacts.context_manifest_csv)
    for row in manifest.to_dict(orient="records"):
        parquet_path = Path(row["parquet_path"])
        if parquet_path.exists():
            tables[str(row["table_name"])] = pd.read_parquet(parquet_path)
    return tables


def build_preprocessed_input_cache(
    *,
    raw_roots: RawDataRoots,
    cache_dir: str | Path,
    province: str = "24",
    overwrite: bool = False,
) -> ProcessedArtifacts:
    """Reuse an existing processed cache when possible, otherwise build it."""
    root = Path(cache_dir)
    if overwrite and root.exists():
        import shutil

        shutil.rmtree(root)
    if not overwrite and (root / "context_manifest.csv").exists() and (root / "seeds").exists():
        return load_processed_artifacts(root)
    return export_processed_inputs(raw_roots, output_dir=root, province=province)
