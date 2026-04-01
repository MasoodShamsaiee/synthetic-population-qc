from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_population_qc import notebook_functions as nf


def _project_root(cwd: Path | None = None) -> Path:
    env_root = os.environ.get("SYNTHPOP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = (cwd or Path.cwd()).resolve()
    return here.parent if here.name.lower() == "notebooks" else here


def run_synthetic_population_pipeline(
    *,
    data_root: str | Path,
    province: str = "24",
    fast: bool = True,
    from_idx: int = 0,
    to_idx: int | None = 25,
    qa_n_das: int = 5,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    save_csv: bool = True,
    save_parquet: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    End-to-end synthetic population generation using local module functions.

    Notes
    -----
    This function does not execute any notebook code at runtime.
    It uses functions copied from `generate_synth_pop.ipynb` into
    `src/synthpop/notebook_functions.py`.
    """
    from tqdm.auto import tqdm

    np.random.seed(int(random_seed))
    random.seed(int(random_seed))

    data_root = Path(data_root)

    # Load inputs.
    df_indiv = nf.load_indiv(data_path=str(data_root), province=province, filtered=True)
    da_codes, place_slug = nf.load_DAs(data_path=str(data_root), province=province)
    census = nf.load_census_profile(data_path=str(data_root))

    (
        total_vb_id,
        total_age_by_sex_vb_id,
        total_hh_vb_id,
        age_vb,
        hdgree_vb,
        lfact_vb,
        hhsize_vb,
        totinc_vb,
        cfstat_vb,
        cfstat_size_vb,
    ) = nf.load_vbs_ids(census)

    # Set shared IDs/config as module globals expected by copied notebook functions.
    nf.total_vb_id = total_vb_id
    nf.total_age_by_sex_vb_id = total_age_by_sex_vb_id
    nf.total_hh_vb_id = total_hh_vb_id
    nf.age_vb = age_vb
    nf.hdgree_vb = hdgree_vb
    nf.lfact_vb = lfact_vb
    nf.hhsize_vb = hhsize_vb
    nf.totinc_vb = totinc_vb
    nf.cfstat_vb = cfstat_vb
    nf.cfstat_size_vb = cfstat_size_vb

    province_census = census.loc[census["geocode"].astype(str) == str(province)].copy()
    seed = nf.load_seed(df_indiv, fast=fast)

    # Keep DAs with positive population.
    da_total_pop = (
        census.loc[census["variableId"] == total_vb_id, ["geocode", "total"]]
        .assign(total=lambda df: pd.to_numeric(df["total"], errors="coerce"))
    )
    valid_das = set(da_total_pop.loc[da_total_pop["total"] > 0, "geocode"].astype(str))
    da_codes = np.array([str(x) for x in da_codes if str(x) in valid_das], dtype=object)

    start = max(0, int(from_idx))
    end = len(da_codes) if to_idx is None else min(len(da_codes), int(to_idx))
    da_slice = da_codes[start:end]

    syn_inds = pd.DataFrame()
    iterator = tqdm(da_slice, desc="Synthesizing DAs") if show_progress else da_slice

    for da_code in iterator:
        da_census = census.loc[census["geocode"].astype(str) == str(da_code)]
        if da_census.empty:
            continue

        # Skip invalid/empty DAs.
        v_total = da_census.loc[da_census["variableId"] == total_vb_id, "total"]
        if v_total.empty or v_total.iloc[0] in ("..", "0"):
            continue

        v_age = da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id, "total"]
        if v_age.empty or pd.isna(v_age.iloc[0]) or float(v_age.iloc[0]) == 0:
            continue

        syn_inds = nf.synthetise_pop_da(
            syn_inds=syn_inds,
            DA_code=da_code,
            da_census=da_census,
            province_census=province_census,
            seed=seed,
            fast=fast,
        )

    # Output paths.
    if output_dir is None:
        output_dir = Path(
            os.environ.get(
                "SYNTHPOP_OUTPUT_DIR",
                os.environ.get("SYNTHPOP_QC_OUTPUT_DIR", str(_project_root() / "data" / "processed" / "synthetic_population")),
            )
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{place_slug}_p{province}_{'fast' if fast else 'qisi'}_{start}_{end}"
    out_csv = out_dir / f"syn_inds_{run_tag}.csv"
    out_parquet = out_dir / f"syn_inds_{run_tag}.parquet"
    out_qa_csv = out_dir / f"qa_summary_{run_tag}.csv"

    if save_csv:
        syn_inds.to_csv(out_csv, index=False)
    else:
        out_csv = None

    if save_parquet:
        syn_inds.to_parquet(out_parquet, index=False)
    else:
        out_parquet = None

    # QA summary on first synthesized DAs.
    qa_rows: list[dict] = []
    synth_das = syn_inds["area"].dropna().unique().tolist()
    for da_code in synth_das[: int(max(0, qa_n_das))]:
        da_int = int(float(da_code))
        ok = False
        for candidate in (da_int, str(da_int)):
            try:
                summary, _ = nf.quality_report_for_da(
                    DA_code=candidate,
                    syn_inds=syn_inds,
                    census=census,
                    province_census=province_census,
                )
                qa_rows.append(summary)
                ok = True
                break
            except Exception:
                continue
        if not ok:
            # Some DAs can have suppressed/missing marginals in source census tables.
            # Skip QA for those DAs without failing the full pipeline run.
            continue
    qa_summary = pd.DataFrame(qa_rows)
    if not qa_summary.empty:
        qa_summary.to_csv(out_qa_csv, index=False)
    else:
        out_qa_csv = None

    return {
        "syn_inds": syn_inds,
        "qa_summary": qa_summary,
        "paths": {"csv": out_csv, "parquet": out_parquet, "qa_csv": out_qa_csv},
        "meta": {
            "province": str(province),
            "fast": bool(fast),
            "from_idx": int(start),
            "to_idx": int(end),
            "n_valid_das": int(len(da_codes)),
            "n_processed_das": int(len(da_slice)),
            "n_synth_inds": int(len(syn_inds)),
            "run_tag": run_tag,
        },
    }
