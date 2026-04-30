# Core Workflow

## What Is Official

The supported workflow in this repository is the bundle-first energy workflow exposed through:

`synthetic_population_qc.run_energy_population_workflow`

This is the main path for:

1. loading raw public inputs
2. exporting processed and harmonized intermediate artifacts
3. planning a support-aware workflow
4. generating household and person outputs
5. validating fit and coherence
6. exporting reusable exploration outputs

The workflow is intended to be Canada-wide. Regional runs are produced by filtering the shared national census, PUMF, and context inputs rather than by maintaining separate city-specific pipelines.

## Inputs

The workflow expects:

- `data_root`
  - shared raw-data repository root or `data/raw` root
  - includes DA-scale census/context tables
  - may also include sibling raw geometry and PUMF folders
- `census_pumf_root`
  - usually `data/raw/PUMF`
  - contains the 2021 Census individual and hierarchical PUMFs
- `housing_survey_root`
  - optional CHS inputs
- `base_population_path`
  - optional pre-generated base population if you are bypassing raw core-population generation
- `da_codes` or `da_codes_file`
  - optional explicit DA scope that overrides named geography selection

## Workflow Steps

The workflow follows this assignment sequence:

1. preprocess and harmonize inputs
2. support assessment and workflow planning
3. core population generation
4. direct household assignment
5. direct person assignment
6. sparse fallback assignment
7. validation and exploration

Household-person coherence remains a design requirement, including:

- `couple_without_children` must be size `2`
- `couple_with_children` must be size `3+`
- `one_parent` must be size `2+`
- `one_person` must be size `1`

Any violations should surface in the household coherence audit.

## Outputs

Every run emits:

- summary and detail fit metrics
- support classification
- sparse-handling report
- assignment-route decision report
- household coherence audit
- reusable Plotly exploration outputs
- DA maps when geometry is available

## Supported Parameters

The supported workflow uses one official public output field per concept.

Core-generation person parameters:

- `sex`
- `age_group`
- `education_level`
- `labour_force_status`
- `household_income`
- `family_status`

Core-generation household parameters:

- `household_size`
- `household_type`

Extension parameters:

- `citizenship_status`
- `immigrant_status`
- `commute_mode`
- `commute_duration`
- `dwelling_type`
- `tenure`
- `bedrooms`
- `period_built`
- `dwelling_condition`
- `core_housing_need`

Internal coded columns such as `agegrp`, `hdgree`, `lfact`, `hhsize`, `totinc`, `cfstat`, and `hhtype` may still appear inside the workflow, but they are not part of the supported output surface.

## Consuming Results

Use `synthetic_population_qc.load_run_bundle` to inspect the standardized bundle from notebooks or downstream code.

Use explicit `da_codes` whenever you want deterministic geographic scope selection. `geography_scope` remains available for convenience and map loading, but explicit DA lists take precedence.
