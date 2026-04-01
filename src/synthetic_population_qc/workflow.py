from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from synthetic_population_qc import notebook_functions as nf
from synthetic_population_qc.households import assign_households_using_census
from synthetic_population_qc.quality import build_quality_summary_table


def _project_root(cwd: Path | None = None) -> Path:
    env_root = os.environ.get("SYNTHPOP_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = (cwd or Path.cwd()).resolve()
    return here.parent if here.name.lower() == "notebooks" else here


def norm_code(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _safe_float_from_census(census_df: pd.DataFrame, variable_id: int) -> float | None:
    s = census_df.loc[census_df["variableId"] == variable_id, "total"]
    if s.empty:
        return None
    v = s.iloc[0]
    if v in ("x", "F", "..", "", None):
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return None if pd.isna(x) else x


def _bucket_totinc(census_df: pd.DataFrame, mapping: dict[int, int]) -> dict[int, float] | None:
    b0 = _safe_float_from_census(census_df, mapping[0])
    b0a = _safe_float_from_census(census_df, mapping[0] + 1)
    b1 = [_safe_float_from_census(census_df, mapping[1] + k) for k in range(4)]
    b2 = [_safe_float_from_census(census_df, mapping[2] + k) for k in range(4)]
    b3 = _safe_float_from_census(census_df, mapping[3])

    if b0 is None or b0a is None or any(x is None for x in b1) or any(x is None for x in b2) or b3 is None:
        return None

    return {0: b0 + b0a, 1: sum(b1), 2: sum(b2), 3: b3}


def _largest_remainder_to_total(values_dict: dict[int, float], target_total: int) -> dict[int, int]:
    keys = list(values_dict.keys())
    floors = {k: int(math.floor(max(0.0, float(values_dict[k])))) for k in keys}
    rem = int(target_total - sum(floors.values()))
    frac = sorted(
        [(k, float(values_dict[k]) - math.floor(float(values_dict[k]))) for k in keys],
        key=lambda t: t[1],
        reverse=True,
    )

    if rem > 0:
        for i in range(rem):
            floors[frac[i % len(frac)][0]] += 1
    elif rem < 0:
        need = -rem
        frac_rev = sorted(frac, key=lambda t: t[1])
        i = 0
        while need > 0 and i < 100000:
            k = frac_rev[i % len(frac_rev)][0]
            if floors[k] > 0:
                floors[k] -= 1
                need -= 1
            i += 1
    return floors


def patch_totinc_loader() -> None:
    def load_marginals_totinc_robust(da_census: pd.DataFrame, province_census: pd.DataFrame, total_pop: int) -> dict[int, int]:
        da_bucket = _bucket_totinc(da_census, nf.totinc_vb)
        if da_bucket is not None:
            return {k: int(v) for k, v in da_bucket.items()}

        prov_bucket = _bucket_totinc(province_census, nf.totinc_vb)
        if prov_bucket is None:
            eq = {0: total_pop / 4.0, 1: total_pop / 4.0, 2: total_pop / 4.0, 3: total_pop / 4.0}
            return _largest_remainder_to_total(eq, int(total_pop))

        prov_sum = sum(prov_bucket.values())
        if prov_sum <= 0:
            eq = {0: total_pop / 4.0, 1: total_pop / 4.0, 2: total_pop / 4.0, 3: total_pop / 4.0}
            return _largest_remainder_to_total(eq, int(total_pop))

        scaled = {k: float(total_pop) * (prov_bucket[k] / prov_sum) for k in prov_bucket}
        return _largest_remainder_to_total(scaled, int(total_pop))

    nf.load_marginals_totinc = load_marginals_totinc_robust


@dataclass
class SynthpopContext:
    data_root: Path
    province: str
    city_scope: str
    fast: bool
    random_seed: int
    out_dir: Path
    place_slug: str
    df_indiv: pd.DataFrame
    census: pd.DataFrame
    province_census: pd.DataFrame
    seed: np.ndarray
    raw_da_codes: list[str]
    census_by_da: dict[str, pd.DataFrame]
    total_vb_id: int
    total_age_by_sex_vb_id: int
    total_hh_vb_id: int
    age_vb: dict[int, int]
    hdgree_vb: dict[int, int]
    lfact_vb: dict[int, int]
    hhsize_vb: dict[int, int]
    totinc_vb: dict[int, int]
    cfstat_vb: dict[int, int]
    cfstat_size_vb: dict[str, int]


@dataclass(frozen=True)
class IncrementalRunPlan:
    run_id: str
    snapshot_tag: str
    sample_label: str
    target_das: list[str]
    completed_checkpoint_das: list[str]
    completed_target_das: list[str]
    remaining_das: list[str]
    checkpoint_prefix: str
    checkpoint_df: pd.DataFrame
    invalid_checkpoint_df: pd.DataFrame


def _load_city_da_list(city_list_dir: Path, name: str) -> set[str]:
    p = city_list_dir / f"da_list_{name}.csv"
    if not p.exists():
        return set()
    d = pd.read_csv(p)
    if "da_code" not in d.columns:
        return set()
    return set(d["da_code"].map(norm_code).dropna().astype(str))


def _city_scope_da_codes(city_scope: str, city_list_dir: Path) -> set[str]:
    city_lists = {
        "montreal": _load_city_da_list(city_list_dir, "montreal"),
        "quebec_city": _load_city_da_list(city_list_dir, "quebec_city"),
        "trois_rivieres": _load_city_da_list(city_list_dir, "trois_rivieres"),
    }
    if city_scope == "all_qc":
        return set()
    if city_scope == "all_three_cities":
        return city_lists["montreal"] | city_lists["quebec_city"] | city_lists["trois_rivieres"]
    if city_scope not in city_lists:
        raise ValueError("city_scope must be one of: all_qc, montreal, quebec_city, trois_rivieres, all_three_cities")
    return city_lists[city_scope]


def _load_city_fsa_codes(project_root: Path, city_name: str) -> set[str]:
    geometry_map = {
        "montreal": "Montreal.geojson",
        "quebec_city": "Quebec_city.geojson",
        "trois_rivieres": "Trois_Rivieres.geojson",
    }
    geojson_name = geometry_map.get(city_name)
    if geojson_name is None:
        return set()

    p = project_root / "data" / "raw" / "geometry" / geojson_name
    if not p.exists():
        return set()

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    fsas: set[str] = set()
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        fsa = props.get("CFSAUID")
        if fsa is not None and str(fsa).strip():
            fsas.add(str(fsa).strip().upper())
    return fsas


def _city_scope_fsa_codes(city_scope: str, project_root: Path) -> set[str]:
    if city_scope == "all_qc":
        return set()
    if city_scope == "all_three_cities":
        return (
            _load_city_fsa_codes(project_root, "montreal")
            | _load_city_fsa_codes(project_root, "quebec_city")
            | _load_city_fsa_codes(project_root, "trois_rivieres")
        )
    return _load_city_fsa_codes(project_root, city_scope)


def build_context(
    *,
    data_root: str | Path,
    province: str = "24",
    city_scope: str = "all_qc",
    city_list_dir: str | Path | None = None,
    fast: bool = True,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
    patch_totinc: bool = True,
    force_rebuild_census_cache: bool = False,
) -> SynthpopContext:
    np.random.seed(int(random_seed))
    random.seed(int(random_seed))

    data_root = Path(data_root)
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(
            os.environ.get(
                "SYNTHPOP_OUTPUT_DIR",
                os.environ.get("SYNTHPOP_QC_OUTPUT_DIR", str(_project_root() / "data" / "processed" / "synthetic_population")),
            )
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    city_list_dir = Path(city_list_dir) if city_list_dir is not None else out_dir
    project_root = _project_root()

    df_indiv = nf.load_indiv(data_path=str(data_root), province=str(province), filtered=True)
    raw_da_codes, place_slug = nf.load_DAs(data_path=str(data_root), province=str(province))
    raw_da_codes = [norm_code(x) for x in raw_da_codes]
    raw_da_codes = [x for x in raw_da_codes if x is not None]

    city_das = _city_scope_da_codes(str(city_scope), city_list_dir)
    city_fsas = _city_scope_fsa_codes(str(city_scope), project_root)

    geocode_filter = raw_da_codes
    cache_path = None
    if str(city_scope) != "all_qc":
        if not city_das:
            raise RuntimeError(f"City DA list for city_scope={city_scope} is missing or empty in {city_list_dir}")
        raw_da_codes = [x for x in raw_da_codes if x in city_das]
        geocode_filter = raw_da_codes
        cache_path = out_dir / f"census_profile_{city_scope}_p{province}.parquet"

    census = nf.load_census_profile(
        data_path=str(data_root),
        province=str(province),
        geocode_filter=geocode_filter,
        fsa_codes=city_fsas,
        cache_path=cache_path,
        force_rebuild_cache=bool(force_rebuild_census_cache),
    )

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

    if patch_totinc:
        patch_totinc_loader()

    census = census.copy()
    census["geocode_norm"] = census["geocode"].map(norm_code)

    province_census = census.loc[census["geocode_norm"] == str(province)].copy()
    seed = nf.load_seed(df_indiv, fast=fast)
    census_by_da = {k: v.copy() for k, v in census.groupby("geocode_norm", sort=False)}

    return SynthpopContext(
        data_root=data_root,
        province=str(province),
        city_scope=str(city_scope),
        fast=bool(fast),
        random_seed=int(random_seed),
        out_dir=out_dir,
        place_slug=str(place_slug).strip().lower(),
        df_indiv=df_indiv,
        census=census,
        province_census=province_census,
        seed=seed,
        raw_da_codes=raw_da_codes,
        census_by_da=census_by_da,
        total_vb_id=int(total_vb_id),
        total_age_by_sex_vb_id=int(total_age_by_sex_vb_id),
        total_hh_vb_id=int(total_hh_vb_id),
        age_vb=age_vb,
        hdgree_vb=hdgree_vb,
        lfact_vb=lfact_vb,
        hhsize_vb=hhsize_vb,
        totinc_vb=totinc_vb,
        cfstat_vb=cfstat_vb,
        cfstat_size_vb=cfstat_size_vb,
    )


def candidate_da_codes(ctx: SynthpopContext) -> list[str]:
    raw_set = set(ctx.raw_da_codes)
    census_geo_set = set(ctx.census["geocode_norm"].dropna().unique().tolist())
    return sorted(raw_set & census_geo_set)


def sample_da_codes(da_codes: Iterable[str], *, sample_ratio: float | None, sample_seed: int = 42) -> list[str]:
    all_codes = sorted(set(da_codes))
    if not all_codes:
        return []
    if sample_ratio is None:
        return all_codes
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be in (0,1] or None")
    n = max(1, int(round(len(all_codes) * sample_ratio)))
    n = min(n, len(all_codes))
    rng = random.Random(int(sample_seed))
    return sorted(rng.sample(all_codes, n))


def sample_label(sample_ratio: float | None) -> str:
    return "all" if sample_ratio is None else f"sample{str(sample_ratio).replace('.', 'p')}"


def target_count(total_codes: int, sample_ratio: float | None) -> int:
    if total_codes <= 0:
        return 0
    if sample_ratio is None:
        return int(total_codes)
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be in (0, 1] or None")
    return int(min(total_codes, max(1, round(total_codes * sample_ratio))))


def build_target_das(
    da_codes: Iterable[str],
    *,
    completed_das: Iterable[str],
    target_sample_ratio: float | None,
    sample_seed: int = 42,
) -> list[str]:
    all_codes = sorted(set(map(str, da_codes)))
    if not all_codes:
        return []

    all_set = set(all_codes)
    completed = [str(x) for x in completed_das if str(x) in all_set]
    completed = list(dict.fromkeys(completed))
    target_n = max(target_count(len(all_codes), target_sample_ratio), len(completed))

    ordered = all_codes.copy()
    rng = random.Random(int(sample_seed))
    rng.shuffle(ordered)

    completed_set = set(completed)
    remaining_pool = [da for da in ordered if da not in completed_set]
    needed = max(0, target_n - len(completed))
    return completed + remaining_pool[:needed]


def discover_checkpoints(
    *,
    out_dir: str | Path,
    checkpoint_prefix: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out_dir = Path(out_dir)
    pattern = re.compile(rf"^{re.escape(checkpoint_prefix)}_(\d+)_(\d+)_(\d+)\.parquet$")
    rows = []
    invalid_rows = []
    completed_das: list[str] = []

    for path in sorted(out_dir.glob(f"{checkpoint_prefix}_*.parquet")):
        match = pattern.match(path.name)
        if not match:
            continue

        batch_id, start, end = map(int, match.groups())
        try:
            area_df = pd.read_parquet(path, columns=["area"])
            area_codes = [norm_code(x) for x in area_df["area"].dropna().tolist()]
            area_codes = [x for x in area_codes if x is not None]
        except Exception as exc:
            invalid_rows.append({"path": str(path), "reason": str(exc)})
            continue

        unique_codes = list(dict.fromkeys(area_codes))
        completed_das.extend(unique_codes)
        rows.append(
            {
                "path": path,
                "batch_id": batch_id,
                "start": start,
                "end": end,
                "n_rows": int(len(area_df)),
                "n_das": int(len(unique_codes)),
            }
        )

    checkpoint_df = pd.DataFrame(rows).sort_values(["batch_id", "start", "end"]) if rows else pd.DataFrame()
    invalid_df = pd.DataFrame(invalid_rows)
    completed_unique = list(dict.fromkeys(completed_das))
    return checkpoint_df, completed_unique, invalid_df


def combine_checkpoint_population(
    checkpoint_df: pd.DataFrame,
    *,
    target_das: Iterable[str],
) -> pd.DataFrame:
    if checkpoint_df.empty:
        return pd.DataFrame()

    frames = [pd.read_parquet(path) for path in checkpoint_df["path"].tolist()]
    syn_inds_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if syn_inds_all.empty:
        return syn_inds_all

    keep = set(map(str, target_das))
    area_norm = syn_inds_all["area"].map(norm_code).astype(str)
    return syn_inds_all.loc[area_norm.isin(keep)].reset_index(drop=True)


def plan_incremental_run(
    *,
    ctx: SynthpopContext,
    run_id: str,
    target_sample_ratio: float | None,
) -> IncrementalRunPlan:
    all_candidate_das = candidate_da_codes(ctx)
    checkpoint_prefix = f"syn_inds_batch_{run_id}"
    checkpoint_df, completed_checkpoint_das, invalid_checkpoint_df = discover_checkpoints(
        out_dir=ctx.out_dir,
        checkpoint_prefix=checkpoint_prefix,
    )
    target_das = build_target_das(
        all_candidate_das,
        completed_das=completed_checkpoint_das,
        target_sample_ratio=target_sample_ratio,
        sample_seed=ctx.random_seed,
    )
    sample_lbl = sample_label(target_sample_ratio)
    snapshot_tag = f"{run_id}_{sample_lbl}"
    completed_set = set(completed_checkpoint_das)
    completed_target_das = [da for da in target_das if da in completed_set]
    remaining_das = [da for da in target_das if da not in completed_set]
    return IncrementalRunPlan(
        run_id=run_id,
        snapshot_tag=snapshot_tag,
        sample_label=sample_lbl,
        target_das=target_das,
        completed_checkpoint_das=completed_checkpoint_das,
        completed_target_das=completed_target_das,
        remaining_das=remaining_das,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_df=checkpoint_df,
        invalid_checkpoint_df=invalid_checkpoint_df,
    )


def run_incremental_snapshot_workflow(
    *,
    ctx: SynthpopContext,
    run_id: str,
    target_sample_ratio: float | None,
    batch_size: int = 200,
    save_batch_checkpoints: bool = True,
    run_quality: bool = True,
    size_5plus: int = 5,
    show_progress: bool = True,
) -> dict:
    plan = plan_incremental_run(ctx=ctx, run_id=run_id, target_sample_ratio=target_sample_ratio)

    resume = {"syn_inds": pd.DataFrame(), "meta_df": pd.DataFrame(), "failed_df": pd.DataFrame()}
    if plan.remaining_das:
        resume = run_synthesis_for_da_codes(
            ctx=ctx,
            da_codes=plan.remaining_das,
            batch_size=batch_size,
            save_batch_checkpoints=save_batch_checkpoints,
            checkpoint_prefix=plan.checkpoint_prefix,
            start_index_offset=len(plan.completed_target_das),
            batch_id_offset=int(plan.checkpoint_df["batch_id"].max()) if not plan.checkpoint_df.empty else 0,
            show_progress=show_progress,
        )

    checkpoint_df, completed_checkpoint_das, invalid_checkpoint_df = discover_checkpoints(
        out_dir=ctx.out_dir,
        checkpoint_prefix=plan.checkpoint_prefix,
    )
    syn_inds_all = combine_checkpoint_population(checkpoint_df, target_das=plan.target_das)

    syn_with_hh = pd.DataFrame()
    hh_assign_summary = pd.DataFrame()
    quality_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    figs = {}

    if not syn_inds_all.empty:
        syn_with_hh, hh_assign_summary = assign_households_using_census(
            syn_inds=syn_inds_all,
            census=ctx.census,
            province=ctx.province,
            area_col="area",
            total_hh_vb_id=ctx.total_hh_vb_id,
            hhsize_vb=ctx.hhsize_vb,
            random_seed=ctx.random_seed,
            size_5plus=size_5plus,
            show_progress=show_progress,
        )

        if run_quality:
            quality_df = build_quality_summary_table(
                syn_with_hh=syn_with_hh,
                census=ctx.census,
                total_vb_id=ctx.total_vb_id,
                total_age_by_sex_vb_id=ctx.total_age_by_sex_vb_id,
                total_hh_vb_id=ctx.total_hh_vb_id,
                hhsize_vb=ctx.hhsize_vb,
                totinc_vb=ctx.totinc_vb,
                cfstat_vb=ctx.cfstat_vb,
                area_col="area",
                show_progress=show_progress,
            )

        from synthetic_population_qc.evaluation import run_eval_suite

        metrics_df, summary_df, figs = run_eval_suite(
            syn_with_hh=syn_with_hh,
            census=ctx.census,
            total_vb_id=ctx.total_vb_id,
            total_age_by_sex_vb_id=ctx.total_age_by_sex_vb_id,
            total_hh_vb_id=ctx.total_hh_vb_id,
            area_col="area",
        )

    selected_path = ctx.out_dir / f"selected_das_{plan.snapshot_tag}.csv"
    syn_path = ctx.out_dir / f"syn_inds_{plan.snapshot_tag}.parquet"
    meta_path = ctx.out_dir / f"batch_meta_{plan.snapshot_tag}.csv"
    failed_path = ctx.out_dir / f"failed_das_{plan.snapshot_tag}.csv"
    syn_hh_path = ctx.out_dir / f"syn_inds_with_hh_{plan.snapshot_tag}.parquet"
    hh_sum_path = ctx.out_dir / f"hh_assignment_summary_{plan.snapshot_tag}.csv"
    quality_path = ctx.out_dir / f"quality_summary_{plan.snapshot_tag}.csv"
    metrics_path = ctx.out_dir / f"eval_da_metrics_{plan.snapshot_tag}.csv"
    summary_path = ctx.out_dir / f"eval_error_summary_{plan.snapshot_tag}.csv"

    pd.DataFrame({"da_code": plan.target_das}).to_csv(selected_path, index=False)
    syn_inds_all.to_parquet(syn_path, index=False)
    resume["meta_df"].to_csv(meta_path, index=False)
    resume["failed_df"].to_csv(failed_path, index=False)
    if not syn_with_hh.empty:
        syn_with_hh.to_parquet(syn_hh_path, index=False)
    if not hh_assign_summary.empty:
        hh_assign_summary.to_csv(hh_sum_path, index=False)
    if run_quality and not quality_df.empty:
        quality_df.to_csv(quality_path, index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_path, index=False)
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)

    return {
        "plan": plan,
        "resume": resume,
        "checkpoint_df": checkpoint_df,
        "completed_checkpoint_das": completed_checkpoint_das,
        "invalid_checkpoint_df": invalid_checkpoint_df,
        "syn_inds_all": syn_inds_all,
        "syn_with_hh": syn_with_hh,
        "hh_assign_summary": hh_assign_summary,
        "quality_df": quality_df,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "figs": figs,
        "paths": {
            "selected_das": selected_path,
            "syn_inds": syn_path,
            "meta": meta_path,
            "failed": failed_path,
            "syn_with_hh": syn_hh_path,
            "hh_summary": hh_sum_path,
            "quality": quality_path,
            "metrics": metrics_path,
            "summary": summary_path,
        },
    }


def run_synthesis_for_da_codes(
    *,
    ctx: SynthpopContext,
    da_codes: list[str],
    batch_size: int = 200,
    save_batch_checkpoints: bool = True,
    checkpoint_prefix: str = "syn_inds_batch",
    start_index_offset: int = 0,
    batch_id_offset: int = 0,
    show_progress: bool = True,
) -> dict:
    from tqdm.auto import tqdm

    all_syn_parts: list[pd.DataFrame] = []
    failed_das: list[dict] = []
    batch_meta: list[dict] = []

    for batch_id, local_start in enumerate(
        range(0, len(da_codes), int(batch_size)),
        start=1 + int(batch_id_offset),
    ):
        local_end = min(local_start + int(batch_size), len(da_codes))
        start = local_start + int(start_index_offset)
        end = local_end + int(start_index_offset)
        da_slice = da_codes[local_start:local_end]
        syn_batch = pd.DataFrame()

        iterator = tqdm(da_slice, desc=f"Synth batch {batch_id}") if show_progress else da_slice
        for da_code in iterator:
            da_census = ctx.census_by_da.get(da_code)
            if da_census is None or da_census.empty:
                continue

            v_age = da_census.loc[da_census["variableId"] == ctx.total_age_by_sex_vb_id, "total"]
            if v_age.empty:
                continue
            try:
                if pd.isna(v_age.iloc[0]) or float(v_age.iloc[0]) == 0:
                    continue
            except Exception:
                continue

            try:
                syn_batch = nf.synthetise_pop_da(
                    syn_inds=syn_batch,
                    DA_code=da_code,
                    da_census=da_census,
                    province_census=ctx.province_census,
                    seed=ctx.seed,
                    fast=ctx.fast,
                )
            except Exception as e:
                failed_das.append({"batch_id": batch_id, "da_code": da_code, "error": str(e)})
                continue

        syn_batch["batch_id"] = batch_id
        syn_batch["batch_start"] = start
        syn_batch["batch_end"] = end
        all_syn_parts.append(syn_batch)

        batch_meta.append(
            {
                "batch_id": batch_id,
                "from_idx": start,
                "to_idx": end,
                "n_das_in_batch": len(da_slice),
                "n_syn_inds_batch": int(len(syn_batch)),
            }
        )

        if save_batch_checkpoints:
            chk_path = ctx.out_dir / f"{checkpoint_prefix}_{batch_id:04d}_{start}_{end}.parquet"
            syn_batch.to_parquet(chk_path, index=False)

    syn_inds = pd.concat(all_syn_parts, ignore_index=True) if all_syn_parts else pd.DataFrame()
    failed_df = pd.DataFrame(failed_das)
    meta_df = pd.DataFrame(batch_meta)
    return {"syn_inds": syn_inds, "failed_df": failed_df, "meta_df": meta_df}


def quality_sample(
    *,
    ctx: SynthpopContext,
    syn_inds: pd.DataFrame,
    qa_n_das: int = 0,
    show_progress: bool = True,
) -> pd.DataFrame:
    from tqdm.auto import tqdm

    if qa_n_das <= 0 or syn_inds.empty:
        return pd.DataFrame()

    qa_rows: list[dict] = []
    synth_das = syn_inds["area"].dropna().map(norm_code).dropna().unique().tolist()[: int(qa_n_das)]
    iterator = tqdm(synth_das, desc="QA sample") if show_progress else synth_das
    for da_code in iterator:
        try_codes = [da_code]
        try:
            try_codes = [int(float(da_code)), str(int(float(da_code)))]
        except Exception:
            pass

        for cand in try_codes:
            try:
                summary, _ = nf.quality_report_for_da(
                    DA_code=cand,
                    syn_inds=syn_inds,
                    census=ctx.census,
                    province_census=ctx.province_census,
                )
                qa_rows.append(summary)
                break
            except Exception:
                continue

    return pd.DataFrame(qa_rows)


def run_full_workflow(
    *,
    ctx: SynthpopContext,
    da_codes: list[str],
    run_tag: str,
    batch_size: int = 200,
    sample_ratio: float | None = None,
    qa_n_das: int = 0,
    save_batch_checkpoints: bool = True,
    assign_households: bool = True,
    run_quality: bool = True,
    size_5plus: int = 5,
    show_progress: bool = True,
) -> dict:
    selected = sample_da_codes(da_codes, sample_ratio=sample_ratio, sample_seed=ctx.random_seed)
    synth = run_synthesis_for_da_codes(
        ctx=ctx,
        da_codes=selected,
        batch_size=batch_size,
        save_batch_checkpoints=save_batch_checkpoints,
        checkpoint_prefix=f"syn_inds_batch_{run_tag}",
        show_progress=show_progress,
    )
    syn_inds = synth["syn_inds"]
    meta_df = synth["meta_df"]
    failed_df = synth["failed_df"]

    qa_summary = quality_sample(ctx=ctx, syn_inds=syn_inds, qa_n_das=qa_n_das, show_progress=show_progress)

    syn_with_hh = pd.DataFrame()
    hh_assign_summary = pd.DataFrame()
    if assign_households and not syn_inds.empty:
        syn_with_hh, hh_assign_summary = assign_households_using_census(
            syn_inds=syn_inds,
            census=ctx.census,
            province=ctx.province,
            area_col="area",
            total_hh_vb_id=ctx.total_hh_vb_id,
            hhsize_vb=ctx.hhsize_vb,
            random_seed=ctx.random_seed,
            size_5plus=size_5plus,
            show_progress=show_progress,
        )

    quality_df = pd.DataFrame()
    if run_quality and not syn_with_hh.empty:
        quality_df = build_quality_summary_table(
            syn_with_hh=syn_with_hh,
            census=ctx.census,
            total_vb_id=ctx.total_vb_id,
            total_age_by_sex_vb_id=ctx.total_age_by_sex_vb_id,
            total_hh_vb_id=ctx.total_hh_vb_id,
            hhsize_vb=ctx.hhsize_vb,
            totinc_vb=ctx.totinc_vb,
            cfstat_vb=ctx.cfstat_vb,
            area_col="area",
            show_progress=show_progress,
        )

    paths = {
        "syn": ctx.out_dir / f"syn_inds_{run_tag}.parquet",
        "qa": ctx.out_dir / f"qa_summary_{run_tag}.csv",
        "meta": ctx.out_dir / f"batch_meta_{run_tag}.csv",
        "failed": ctx.out_dir / f"failed_das_{run_tag}.csv",
        "selected_das": ctx.out_dir / f"selected_das_{run_tag}.csv",
        "syn_with_hh": ctx.out_dir / f"syn_inds_with_hh_{run_tag}.parquet",
        "hh_summary": ctx.out_dir / f"hh_assignment_summary_{run_tag}.csv",
        "quality": ctx.out_dir / f"quality_summary_{run_tag}.csv",
    }

    syn_inds.to_parquet(paths["syn"], index=False)
    qa_summary.to_csv(paths["qa"], index=False)
    meta_df.to_csv(paths["meta"], index=False)
    failed_df.to_csv(paths["failed"], index=False)
    pd.DataFrame({"da_code": selected}).to_csv(paths["selected_das"], index=False)

    if not syn_with_hh.empty:
        syn_with_hh.to_parquet(paths["syn_with_hh"], index=False)
    if not hh_assign_summary.empty:
        hh_assign_summary.to_csv(paths["hh_summary"], index=False)
    if not quality_df.empty:
        quality_df.to_csv(paths["quality"], index=False)

    return {
        "syn_inds": syn_inds,
        "qa_summary": qa_summary,
        "meta_df": meta_df,
        "failed_df": failed_df,
        "syn_with_hh": syn_with_hh,
        "hh_assign_summary": hh_assign_summary,
        "quality_df": quality_df,
        "selected_das": selected,
        "paths": paths,
    }
