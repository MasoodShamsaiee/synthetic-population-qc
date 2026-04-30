# synthetic-population-qc

`synthetic-population-qc` now centers on one supported workflow:

`raw public inputs -> processed/harmonized artifacts -> synthesis -> validation -> exploration bundle`

The official workflow is the bundle-first energy workflow for Canadian DA-scale synthetic population generation, with city or province scope handled by filtering the shared national inputs.

## Official Workflow

The official entry point is `synthetic_population_qc.run_energy_population_workflow`.

It accepts:

- the shared raw-data repository root through `data_root`
  - expected layout includes `data/raw/census/...` and typically `data/raw/PUMF/...`
- a Census PUMF root, usually `data/raw/PUMF`
- optional CHS root when that dataset exists separately
- optional geometry root and optional base population path
- optional explicit `da_codes`, `da_codes_file`, or named DA scope selection
- configurable public age grouping through `age_group_scheme` or `age_group_breaks`

It returns a typed `SyntheticPopulationRun` object and writes a standardized run bundle with:

- `manifests/`
- `processed/`
- `synthesis/`
- `validation/`
- `exploration/`

## Quick Start

```python
from synthetic_population_qc import RawDataRoots, WorkflowSettings, run_energy_population_workflow

run = run_energy_population_workflow(
    raw_data=RawDataRoots.from_paths(
        data_root="../urban-energy-data",
        census_pumf_root="../urban-energy-data/data/raw/PUMF",
    ),
    settings=WorkflowSettings.from_paths(
        output_root="data/processed/synthetic_population",
        run_name="canada_scope_smoke",
        da_codes=["24660001"],
    ),
)
```

The returned object exposes paths for processed seeds and context tables, the workflow plan, people and households outputs, validation reports, and exploration artifacts.

When `da_codes` is supplied, it is treated as the authoritative scope selection for the run. `geography_scope` remains available as a convenience selector for map-backed named areas.

## Supported Output Schema

The supported public outputs are intentionally de-duplicated so each concept appears once.

`synthesis/people.parquet` exposes one official column per concept:

- `area`, `HID`
- `sex`, `age_group`, `education_level`, `labour_force_status`, `household_income`, `family_status`
- `household_size`, `household_type`
- `person_uid`
- `citizenship_status`, `immigrant_status`, `commute_mode`, `commute_duration`

Only the official semantic columns above are part of the supported public output contract.

`synthesis/households.parquet` exposes:

- `area`, `household_id`
- `household_size`, `household_type`
- `dwelling_type`, `tenure`, `bedrooms`, `period_built`, `dwelling_condition`, `core_housing_need`

## Package Layout

```text
src/synthetic_population_qc/
  core/       typed run settings and raw-root contracts
  ingest/     preprocessing and harmonized input exports
  synth/      planning and workflow orchestration
  validate/   support and sparse-handling helpers
  explore/    reusable plots and optional DA maps
  runs/       standardized bundle models and loaders
  energy_workflow.py  neutral end-to-end workflow implementation
  context_tables.py  labeled DA-scale census context loaders
  reporting.py       compact fit summaries for generated results
  seed_transforms.py seed harmonization transforms
  utils.py           shared code normalization and reference helpers
```

## Run Bundle

Each run is written to a predictable bundle. Intermediate cache naming is intentionally internal to the workflow; downstream tools should consume the standardized bundle paths instead of cache filenames.

```text
<run_root>/
  manifests/
    bundle_manifest.json
    metadata.json
    workflow_plan.json
    da_coverage.csv
    parameter_mapping_audit.csv
  processed/
    preprocessing_audit.csv
    input_contract.csv
    seeds/
    context/
    context_manifest.csv
  synthesis/
    people.parquet
    households.parquet
  validation/
    summary_metrics.csv
    detail_metrics.csv
    support_classification.csv
    sparse_handling_report.csv
    assignment_route_decisions.csv
    household_coherence_audit.csv
  exploration/
    *.html
    exploration_manifest.json
```

Downstream tools should consume this bundle instead of guessing file names or relying on private intermediate artifacts.

## Environment

Create a fresh local environment from [environment.yml](environment.yml).

Large raw data can still live outside the repo. The workflow accepts explicit paths for census, PUMF, CHS, and optional geometry roots, so the software environment is reproducible even when the datasets are stored separately.

## Docs

- [docs/core_workflow.md](docs/core_workflow.md)
- [docs/data_contracts.md](docs/data_contracts.md)
