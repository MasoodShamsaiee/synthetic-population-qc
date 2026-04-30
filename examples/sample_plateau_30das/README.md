# Sample Plateau 30 DAs

This example package is a clean, local workflow sample built from the Plateau 30-DA subset packaged in this repository.

It is designed to be runnable without reaching back into `examples/sample/` or depending on a messy history of reruns.

## Contents

- `raw/`
  - packaged 30-DA census subset
  - packaged Quebec-filtered Census PUMF subset
  - DA code list and raw manifest
- `processed_input_cache/`
  - harmonized seed and context cache built from the packaged raw subset
- `base_population_cache/`
  - cached core population artifacts for the same 30 DAs
- `workflow_run/`
  - one complete standardized workflow bundle
- `da_codes.txt`
  - authoritative DA selection for this example
- `demo_notebook.ipynb`
  - lightweight notebook for inspecting and rerunning the example

## Workflow

The bundle was generated with the official entry point:

```python
run_energy_population_workflow(
    raw_data=RawDataRoots.from_paths(
        data_root=sample_root,
        census_pumf_root=sample_root / "raw" / "PUMF",
    ),
    settings=WorkflowSettings.from_paths(
        output_root=sample_root,
        run_name="workflow_run",
        da_codes=da_codes,
        show_progress=False,
        generate_exploration=True,
        overwrite=False,
    ),
    processed_cache=sample_root / "processed_input_cache",
    base_population_cache=(sample_root / "base_population_cache" / "syn_inds_with_hh_core_population.parquet"),
)
```

## Current bundle

- run root: `examples/sample_plateau_30das/workflow_run`
- people shape: `15546 x 15`
- households shape: `8616 x 10`
