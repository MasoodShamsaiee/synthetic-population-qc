"""Load labeled DA-scale census context tables used by the workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from synthetic_population_qc.utils import norm_code


DA_TABLE_SPECS: dict[str, tuple[str, str, str]] = {
    "age_sex_core": ("age, sex", "fMyRmd2RDbkDqUm_data.csv", "New Text Document.txt"),
    "dwelling_characteristics": ("dwelling char", "V818MNIvbF_data.csv", "New Text Document.txt"),
    "housing": ("housing", "GxvbVaVRu_data.csv", "New Text Document.txt"),
    "commute": ("commute", "ASmEBfq1h9pC_data.csv", "New Text Document.txt"),
    "immigration_citizenship": ("imm, citiz", "iilmxSwOn_data.csv", "New Text Document.txt"),
    "income_detailed": ("income", "tEVGMRS0SSexK_data.csv", "New Text Document.txt"),
    "education_detailed": ("education", "udDo1DFAvI_data.csv", "New Text Document.txt"),
    "labour_detailed": ("labour", "EVtduVuRucOlsLjw_data.csv", "New Text Document.txt"),
    "household_type_size_detailed": ("hh type, size", "sl5brrkpJ5_data.csv", "New Text Document.txt"),
}


def resolve_raw_data_root(data_root: str | Path) -> Path:
    """Resolve the shared raw-data root from a repo root or raw/ directory."""
    root = Path(data_root)
    candidates = (
        root,
        root / "raw",
        root / "data" / "raw",
    )
    for candidate in candidates:
        if (candidate / "census").exists() or (candidate / "PUMF").exists() or (candidate / "geometry").exists():
            return candidate
    for candidate in reversed(candidates):
        if candidate.exists():
            return candidate
    return candidates[-1]


def resolve_da_census_root(data_root: str | Path) -> Path:
    """Resolve the DA-scale census root from a repo root, raw/ root, or data/raw root."""
    raw_root = resolve_raw_data_root(data_root)
    candidates = (raw_root / "census" / "DA scale",)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw_root / "census" / "DA scale"


def resolve_pumf_root(
    data_root: str | Path,
    *,
    census_pumf_root: str | Path | None = None,
) -> Path:
    """Resolve the Census PUMF root from an explicit path or the shared raw tree."""
    if census_pumf_root is None:
        root = Path(data_root)
        if (
            (root / "ind").exists()
            or (root / "heir").exists()
            or (root / "data_donnees_2021_ind_v2.csv").exists()
            or (root / "data_donnees_2021_hier_v2.csv").exists()
        ):
            return root
        raw_root = resolve_raw_data_root(root)
        return raw_root / "PUMF"
    root = Path(census_pumf_root)
    if (
        (root / "ind").exists()
        or (root / "heir").exists()
        or (root / "data_donnees_2021_ind_v2.csv").exists()
        or (root / "data_donnees_2021_hier_v2.csv").exists()
    ):
        return root
    candidates = (
        root,
        root / "PUMF",
        root / "raw" / "PUMF",
        root / "data" / "raw" / "PUMF",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_da_census_metadata(metadata_path: str | Path) -> dict[str, str]:
    """Parse the Statistics Canada metadata file into a column rename map."""
    metadata_path = Path(metadata_path)
    mapping: dict[str, str] = {}
    pattern = re.compile(r"^(COL\d+)\s*-\s*(.+)$")
    for line in metadata_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            mapping[match.group(1)] = match.group(2).strip()
    return mapping


def load_labeled_da_census_extract(csv_path: str | Path, metadata_path: str | Path) -> pd.DataFrame:
    """Read one DA extract and apply human-readable labels plus `da_code`."""
    csv_path = Path(csv_path)
    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for encoding in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if df is None:
        if last_error is not None:
            raise last_error
        df = pd.read_csv(csv_path)

    renamed = df.rename(columns=parse_da_census_metadata(metadata_path))
    geocode_col = next((col for col in renamed.columns if col.startswith("GEO UID")), None)
    if geocode_col is not None:
        renamed["da_code"] = renamed[geocode_col].map(norm_code)
    for col in renamed.columns:
        if col == "da_code" or renamed[col].dtype == object:
            continue
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")
    return renamed


def _resolve_da_table_files(
    raw_root: Path,
    subdir: str,
    expected_csv_name: str,
    metadata_name: str,
) -> tuple[Path | None, Path | None]:
    """Find the CSV and metadata files for one expected DA table folder."""
    dir_path = raw_root / subdir
    meta_path = dir_path / metadata_name
    if not dir_path.exists() or not meta_path.exists():
        return None, None
    expected = dir_path / expected_csv_name
    if expected.exists():
        return expected, meta_path
    candidates = sorted(dir_path.glob("*_data.csv"))
    if candidates:
        return candidates[0], meta_path
    return None, None


def resolve_da_table_artifacts(data_root: str | Path) -> dict[str, tuple[Path | None, Path | None]]:
    """Resolve the CSV/metadata pair for each supported DA census table."""
    raw_root = resolve_da_census_root(data_root)
    return {
        name: _resolve_da_table_files(raw_root, subdir, expected_csv_name, metadata_name)
        for name, (subdir, expected_csv_name, metadata_name) in DA_TABLE_SPECS.items()
    }


def load_context_tables(data_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load every supported DA-scale census table from the declared data root."""
    tables: dict[str, pd.DataFrame] = {}
    for name, (csv_path, meta_path) in resolve_da_table_artifacts(data_root).items():
        tables[name] = (
            load_labeled_da_census_extract(csv_path, meta_path)
            if csv_path is not None and meta_path is not None
            else pd.DataFrame()
        )
    return tables
