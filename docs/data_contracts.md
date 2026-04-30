# Data Contracts

## Purpose

The main workflow is now bundle-first. Contracts are defined around raw inputs, processed artifacts, synthesis outputs, and validation/exploration deliverables rather than notebook state.

## Raw Input Contract

The supported workflow expects:

- a shared raw-data tree rooted at `data_root`, typically with `data/raw/...`
- DA-scale census/context tables under `data/raw/census/DA scale`
- individual and hierarchical Census PUMFs under `data/raw/PUMF` or an explicitly supplied `census_pumf_root`
- optional CHS input under an explicitly supplied `housing_survey_root`
- optional geometry under `geometry_root`

The preprocessing layer is responsible for normalizing DA codes, carrying forward explicit source paths, and writing processed artifacts for reproducibility.

These contracts are intended to support Canada-wide runs. Geography-specific workflows should narrow scope by filtering inputs, not by introducing alternate bundle shapes or parallel artifact conventions.

## Processed Artifact Contract

Every supported run should produce:

- `processed/input_contract.csv`
- `processed/preprocessing_audit.csv`
- `processed/seeds/person_seed.parquet`
- `processed/seeds/household_seed_hierarchical.parquet`
- `processed/context/*.parquet`
- `processed/context_manifest.csv`

Operational expectations:

- one row per person in the person seed
- one row per donor household in the household seed
- harmonized context tables preserve or create a normalized `da_code`
- processed artifacts are notebook-consumable without hidden path guessing

## Workflow Plan Contract

Every supported run should produce:

- `manifests/workflow_plan.json`
- `manifests/da_coverage.csv`
- `manifests/parameter_mapping_audit.csv`
- `validation/support_classification.csv`

The plan must record:

- selected DA scope
- workflow steps
- core-generation attributes
- direct household-assignment attributes
- direct person-assignment attributes
- sparse-fallback attributes
- full workflow attribute inventory
- fallback ladders and support diagnostics
- an auditable parameter/value mapping table covering renames, collapses, and dropped codes

## Synthesis Output Contract

The main synthesis bundle should expose:

- `synthesis/people.parquet`
- `synthesis/households.parquet`

Operational expectations:

- one row per synthetic person in `people.parquet`
- one row per synthetic household in `households.parquet`
- both outputs carry DA identity through the run
- household coherence semantics are auditable from the household table
- `people.parquet` uses the supported semantic columns rather than duplicate alias fields
- `households.parquet` uses the supported household size/type labels as the public representation

Supported `people.parquet` fields:

- `area`, `HID`
- `sex`, `age_group`, `education_level`, `labour_force_status`, `household_income`, `family_status`
- `household_size`, `household_type`
- `person_uid`
- `citizenship_status`, `immigrant_status`, `commute_mode`, `commute_duration`

Supported `households.parquet` fields:

- `area`, `household_id`
- `household_size`, `household_type`
- `dwelling_type`, `tenure`, `bedrooms`, `period_built`, `dwelling_condition`, `core_housing_need`

Unsupported alias examples:

- `hhsize_core` as a duplicate of `household_size`
- downstream code should rely on the supported semantic columns, not internal aliases
- `agegrp`, `hdgree`, `lfact`, `hhsize`, `totinc`, `cfstat`, `hhtype` as internal coded aliases in final public outputs

## Validation Contract

Every supported run should produce:

- `validation/summary_metrics.csv`
- `validation/detail_metrics.csv`
- `validation/support_classification.csv`
- `validation/sparse_handling_report.csv`
- `validation/assignment_route_decisions.csv`
- `validation/household_coherence_audit.csv`

These are first-class workflow outputs, not notebook-only side artifacts.

Validation expectations:

- every supported workflow attribute appears in `summary_metrics.csv` and `detail_metrics.csv`
- core-generation attributes are evaluated against the preserved base-population distribution
- extension attributes are evaluated against DA-scale census/context marginals where available
- validation inventory should match the workflow plan inventory after output de-duplication

## Exploration Contract

Every supported run should produce reusable exploration artifacts under `exploration/`, including:

- method-performance plots
- conditional-distribution plots
- DA map outputs when geometry is available
- `exploration/exploration_manifest.json`

## Scope Contract

The supported repository scope is the workflow itself:

- raw-input discovery and preprocessing
- support-aware synthesis
- validation and exploration exports
