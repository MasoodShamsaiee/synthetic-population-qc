"""Typed runtime contracts shared by the bundle-oriented workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_population_qc.config import default_geometry_dir, default_output_dir


@dataclass(frozen=True)
class RawDataRoots:
    """Paths to the raw/shared inputs needed by the workflow."""
    data_root: Path
    census_pumf_root: Path
    housing_survey_root: Path | None = None
    geometry_root: Path | None = None
    base_population_path: Path | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        data_root: str | Path,
        census_pumf_root: str | Path,
        housing_survey_root: str | Path | None = None,
        geometry_root: str | Path | None = None,
        base_population_path: str | Path | None = None,
    ) -> "RawDataRoots":
        return cls(
            data_root=Path(data_root),
            census_pumf_root=Path(census_pumf_root),
            housing_survey_root=Path(housing_survey_root) if housing_survey_root is not None else None,
            geometry_root=Path(geometry_root) if geometry_root is not None else None,
            base_population_path=Path(base_population_path) if base_population_path is not None else None,
        )

    def resolved_geometry_root(self) -> Path:
        return self.geometry_root if self.geometry_root is not None else default_geometry_dir()


@dataclass(frozen=True)
class WorkflowSettings:
    """Runtime settings for bundle-oriented synthetic population workflow runs."""
    output_root: Path = default_output_dir()
    run_name: str = "energy_workflow_run"
    method: str = "joint_ipu_v1"
    province: str = "24"
    geography_scope: str = "montreal"
    da_scope_name: str | None = None
    da_codes_file: Path | None = None
    random_seed: int = 42
    max_das: int | None = None
    da_codes: tuple[str, ...] | None = None
    age_group_scheme: str = "default_15"
    age_group_breaks: tuple[int, ...] | None = None
    show_progress: bool = True
    generate_exploration: bool = True
    overwrite: bool = True

    @classmethod
    def from_paths(
        cls,
        *,
        output_root: str | Path | None = None,
        run_name: str = "energy_workflow_run",
        method: str = "joint_ipu_v1",
        province: str = "24",
        geography_scope: str = "montreal",
        da_scope_name: str | None = None,
        da_codes_file: str | Path | None = None,
        random_seed: int = 42,
        max_das: int | None = None,
        da_codes: list[str] | tuple[str, ...] | None = None,
        age_group_scheme: str = "default_15",
        age_group_breaks: list[int] | tuple[int, ...] | None = None,
        show_progress: bool = True,
        generate_exploration: bool = True,
        overwrite: bool = True,
    ) -> "WorkflowSettings":
        return cls(
            output_root=Path(output_root) if output_root is not None else default_output_dir(),
            run_name=run_name,
            method=method,
            province=province,
            geography_scope=geography_scope,
            da_scope_name=da_scope_name,
            da_codes_file=Path(da_codes_file) if da_codes_file is not None else None,
            random_seed=random_seed,
            max_das=max_das,
            da_codes=tuple(str(code).strip() for code in da_codes) if da_codes is not None else None,
            age_group_scheme=age_group_scheme,
            age_group_breaks=tuple(int(x) for x in age_group_breaks) if age_group_breaks is not None else None,
            show_progress=show_progress,
            generate_exploration=generate_exploration,
            overwrite=overwrite,
        )

    def resolved_geography_scope(self) -> str:
        """Return the named geography scope used for geometry and default DA selection."""
        return str(self.geography_scope)
