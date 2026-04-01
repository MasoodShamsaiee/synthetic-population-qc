# synthetic-population-qc

`synthetic-population-qc` is the extracted synthetic population package from the larger research codebase. It contains Quebec-oriented synthetic population generation, household assignment, incremental workflow utilities, and DA-level quality/evaluation helpers.

## What is included

- synthetic individual generation pipeline
- notebook-derived census and microsample loading helpers
- incremental snapshot workflow utilities
- household assignment and household-type derivation
- DA-level quality summary tables
- evaluation plots and DA geometry mapping helpers

## Package layout

```text
src/synthetic_population_qc/
  notebook_functions.py
  pipeline.py
  workflow.py
  households.py
  quality.py
  evaluation.py
```

## Quick start

```powershell
conda run -n dsm_qc python -m pip install -e .[dev]
conda run -n dsm_qc python -c "import synthetic_population_qc; print('ok')"
```

## Notes

- `notebook_functions.py` is a moduleized carry-over from the original notebook workflow and still assumes the project’s census/PUMF data layout.
- see [docs/data_contracts.md](docs/data_contracts.md) for the main table assumptions used across the workflow.
- this repo is extractable and installable, but the next round of cleanup should still reduce global state in the notebook-derived code.

## Citation

If you use this repository in research, software, or derivative work, please preserve the license notices and cite the project using [CITATION.cff](CITATION.cff).
