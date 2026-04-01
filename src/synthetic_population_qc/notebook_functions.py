from __future__ import annotations
import math
import os
import random
from pathlib import Path

import humanleague
import numpy as np
import pandas as pd
import pyreadstat


def _default_data_path() -> str:
    return os.environ.get(
        "SYNTHPOP_DATA_ROOT",
        os.environ.get("SYNTHPOP_QC_DATA_ROOT", "./data"),
    )



# ---- from generate_synth_pop.ipynb cell 2 ----
def load_indiv(
    data_path=None,
    rest_of_path="/PUMF",
    file_name="/cen_ind_2021_pumf_v2.dta",
    province="24",
    filtered = True
):
    """
    Load individual-level PUMF data and filter by province.
    """
    data_path = _default_data_path() if data_path is None else data_path
    dtafile = data_path + rest_of_path + file_name
    if filtered:
        usecols = [
            "ppsort", "weight", "agegrp", "Gender",
            "hdgree", "lfact", "TotInc",
            "hhsize", "cfstat", "prihm",
            "cma", "pr"]
        df_indiv, _ = pyreadstat.read_dta(
        dtafile,
        usecols=usecols,
        )
    else:
        df_indiv, _ = pyreadstat.read_dta(
        dtafile,)

    df_indiv["pr"] = df_indiv["pr"].astype(str)

    if province in {"60", "61", "62"}:
        df_indiv = df_indiv.loc[df_indiv["pr"].str.strip() == "70"]
    else:
        df_indiv = df_indiv.loc[df_indiv["pr"].str.strip() == province]

    return df_indiv


# ---- from generate_synth_pop.ipynb cell 6 ----
def load_DAs(
    data_path=None,
    rest_of_path="/DA labels",
    file_name="/2021_92-151_X.csv",
    province="24",
):
    """
    Load DA codes and province name from Census lookup table.
    """
    data_path = _default_data_path() if data_path is None else data_path
    lookup = pd.read_csv(
        data_path + rest_of_path + file_name,
        encoding="ISO-8859-1",
        low_memory=False,
    )

    lookup["pr"] = lookup["PRUID_PRIDU"].astype(str)
    filtered = lookup.loc[lookup["pr"].str.strip() == province]

    place = filtered.iloc[0]["PRENAME_PRANOM"]
    filename = place.replace(" ", "_").lower()

    da_codes = filtered["DAUID_ADIDU"].unique()
    da_codes.sort()

    print(place)
    print(f"{da_codes.size} DAs")

    return da_codes, filename


# ---- from generate_synth_pop.ipynb cell 9 ----
#helper functions

# Map ages to 18 classes
def map_age_grp(df_indiv):
    for i in range(17, 22):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 8
    for i in range(16, 7, -1):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 7
    df_indiv.loc[df_indiv["agegrp"] == 1, "agegrp"] = 10
    df_indiv.loc[df_indiv["agegrp"] == 2, "agegrp"] = 11
    df_indiv.loc[df_indiv["agegrp"] == 3, "agegrp"] = 11
    df_indiv.loc[df_indiv["agegrp"] == 4, "agegrp"] = 12
    df_indiv.loc[df_indiv["agegrp"] == 5, "agegrp"] = 12
    df_indiv.loc[df_indiv["agegrp"] == 6, "agegrp"] = 14
    df_indiv.loc[df_indiv["agegrp"] == 7, "agegrp"] = 14
    df_indiv = df_indiv.loc[df_indiv["agegrp"] != 88]
    return df_indiv


# Map ages to 7 classes
def map_age_grp_new(df_indiv):
    for i in range(17, 22):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = i + 8

    for i in range(16, 7, -1):
        df_indiv.loc[df_indiv["agegrp"] == i, "agegrp"] = 13

    df_indiv.loc[df_indiv["agegrp"] == 1, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 2, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 3, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 4, "agegrp"] = 9
    df_indiv.loc[df_indiv["agegrp"] == 5, "agegrp"] = 9

    df_indiv.loc[df_indiv["agegrp"] == 6, "agegrp"] = 13
    df_indiv.loc[df_indiv["agegrp"] == 7, "agegrp"] = 13
    df_indiv = df_indiv.loc[df_indiv["agegrp"] != 88]
    return df_indiv


# ---- from generate_synth_pop.ipynb cell 10 ----
# Map hdgree to 4 classes
def map_hdgree(df_indiv):
    df_indiv.loc[df_indiv["hdgree"] == 88, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] == 99, "hdgree"] = 1
    df_indiv.loc[df_indiv["hdgree"] > 2, "hdgree"] = 1686
    df_indiv.loc[df_indiv["hdgree"] == 1, "hdgree"] = 1684
    df_indiv.loc[df_indiv["hdgree"] == 2, "hdgree"] = 1685
    return df_indiv


# Map lfact to 3 classes
def map_lfact(df_indiv):
    df_indiv.loc[df_indiv["lfact"] == 1, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] == 2, "lfact"] = 1867
    df_indiv.loc[df_indiv["lfact"] < 11, "lfact"] = 1868
    df_indiv.loc[df_indiv["lfact"] < 100, "lfact"] = 1869
    return df_indiv


# Map hhsize to 5 classes
def map_hhsize(df_indiv):
    df_indiv.loc[df_indiv["hhsize"] == 8, "hhsize"] = 1
    df_indiv.loc[df_indiv["hhsize"] > 5, "hhsize"] = 5
    return df_indiv


# Map totinc to 4 classes
def map_totinc(df_indiv):
    df_indiv = df_indiv.loc[df_indiv["TotInc"] != 88888888]
    df_indiv.loc[df_indiv["TotInc"] == 99999999, "TotInc"] = 695
    df_indiv.loc[df_indiv["TotInc"] < 20000, "TotInc"] = 695

    # for i in range(1, 10):
    #    df_indiv.loc[((df_indiv["TotInc"] >= 10000 * i) & (df_indiv["TotInc"] < 10000 * (i + 1))), "TotInc"] = 695 + i

    df_indiv.loc[((df_indiv["TotInc"] >= 20000) & (df_indiv["TotInc"] < 60000)), "TotInc"] = 697
    df_indiv.loc[((df_indiv["TotInc"] >= 60000) & (df_indiv["TotInc"] < 100000)), "TotInc"] = 701
    df_indiv.loc[df_indiv["TotInc"] >= 100000, "TotInc"] = 705

    return df_indiv

# added a map for prihm
def map_prihm(df_indiv):
    """
    Collapse PUMF prihm into binary:
      1 = household maintainer
      0 = not household maintainer (includes 'not applicable')
    """
    df_indiv.loc[df_indiv["prihm"] == 9, "prihm"] = 0
    return df_indiv


# ---- from generate_synth_pop.ipynb cell 12 ----
# updated map_cfstat below:
def map_cfstat(df_indiv, src_col="cfstat", dst_col="cfstat"):
    """
    Map PUMF person-role CFSTAT codes (1..8) into seed-safe household categories
    that are guaranteed to exist in PUMF and are suitable for IPF.

    Output CFSTAT codes (seed level):
      0 = Couple without children
      1 = Couple with children
      2 = One-parent family
      3 = Non-census-family household (roommates / relatives)
      4 = One-person household

    Rich household types (multigenerational, multiple families, etc.)
    MUST be derived later from constructed households.
    """

    if dst_col != src_col:
        df_indiv[dst_col] = df_indiv[src_col]

    mapping = {
        1: 0,  # married/common-law partner without children
        2: 1,  # married/common-law partner with children
        4: 1,  # child of a couple
        3: 2,  # parent in one-parent family
        5: 2,  # child in one-parent family
        7: 3,  # not in census family, with non-relatives only
        8: 3,  # not in census family, with other relatives
        6: 4,  # person living alone
    }

    df_indiv[dst_col] = df_indiv[dst_col].map(mapping)

    return df_indiv


# ---- from generate_synth_pop.ipynb cell 13 ----
def unlistify(table, columns, sizes, values):
    """
    Converts an n-column table of counts into an n-dimensional array of counts
    """
    pivot = table.pivot_table(index=columns, values=values, aggfunc='sum')
    # order must be same as column order above
    array = np.zeros(sizes, dtype=int)
    array[tuple(pivot.index.codes)] = pivot.values.flat
    return array


# ---- from generate_synth_pop.ipynb cell 15 ----
def get_impossible(seed):
    """
    Build a constraint tensor (same shape as seed) with 1 for allowed states and 0 for impossible ones.
    Dimensions assumed: [sex, prihm, agegrp, hdgree, lfact, hhsize, totinc, cfstat]
    """

    constraints = np.ones(seed.shape)  # start by allowing every state (will zero-out impossible ones)

    # -------------------------
    # Original constraints kept
    # -------------------------

    # (1) Household maintainer (prihm=1) cannot be in the youngest age groups (agegrp=0,1,2)
    constraints[:, 1, 0, :, :, :, :, :] = 0  # no maintainer in agegrp 0
    constraints[:, 1, 1, :, :, :, :, :] = 0  # no maintainer in agegrp 1
    constraints[:, 1, 2, :, :, :, :, :] = 0  # no maintainer in agegrp 2

    # (2) Educational attainment (hdgree > 0) cannot occur in the youngest age groups (agegrp=0,1,2)
    constraints[:, :, 0, 1, :, :, :, :] = 0  # agegrp 0 cannot have hdgree category 1
    constraints[:, :, 0, 2, :, :, :, :] = 0  # agegrp 0 cannot have hdgree category 2
    constraints[:, :, 1, 1, :, :, :, :] = 0  # agegrp 1 cannot have hdgree category 1
    constraints[:, :, 1, 2, :, :, :, :] = 0  # agegrp 1 cannot have hdgree category 2
    constraints[:, :, 2, 1, :, :, :, :] = 0  # agegrp 2 cannot have hdgree category 1
    constraints[:, :, 2, 2, :, :, :, :] = 0  # agegrp 2 cannot have hdgree category 2

    # (3) Labour force: employed/unemployed cannot occur in the youngest age groups (agegrp=0,1,2)
    # Here lfact indices 0 and 1 represent "employed/unemployed" (per your original code logic)
    constraints[:, :, 0, :, 0, :, :, :] = 0  # agegrp 0 cannot be employed
    constraints[:, :, 0, :, 1, :, :, :] = 0  # agegrp 0 cannot be unemployed
    constraints[:, :, 1, :, 0, :, :, :] = 0  # agegrp 1 cannot be employed
    constraints[:, :, 1, :, 1, :, :, :] = 0  # agegrp 1 cannot be unemployed
    constraints[:, :, 2, :, 0, :, :, :] = 0  # agegrp 2 cannot be employed
    constraints[:, :, 2, :, 1, :, :, :] = 0  # agegrp 2 cannot be unemployed
    # (4) 1-person household (hhsize=0) cannot contain children (agegrp=0,1,2)
    constraints[:, :, 0, :, :, 0, :, :] = 0  # no children in 1-person household
    constraints[:, :, 1, :, :, 0, :, :] = 0
    constraints[:, :, 2, :, :, 0, :, :] = 0

    # (5) If hhsize=0 (1-person household), the person must be the household maintainer (prihm=1)
    constraints[:, 0, :, :, :, 0, :, :] = 0  # non-maintainer cannot be in 1-person household

    # (6) Income: children (agegrp=0,1,2) cannot be in non-zero income brackets (totinc=1..3)
    for i in range(1, 4):
        constraints[:, :, 0, :, :, :, i, :] = 0
        constraints[:, :, 1, :, :, :, i, :] = 0
        constraints[:, :, 2, :, :, :, i, :] = 0

    # (7) Children (agegrp=0,1,2) cannot be household maintainer (prihm=1)
    constraints[:, 1, 0, :, :, :, :, :] = 0  # agegrp 0 cannot be prihm
    constraints[:, 1, 1, :, :, :, :, :] = 0  # agegrp 1 cannot be prihm
    constraints[:, 1, 2, :, :, :, :, :] = 0  # agegrp 2 cannot be prihm
    # -------------------------------------------
    # CFSTAT constraints (UPDATED – MINIMAL & SAFE)
    # -------------------------------------------

    # No hard CFSTAT ↔ household-size constraints anymore.
    # CFSTAT is now a coarse, PUMF-supported category used ONLY for composition,
    # not to encode household size or structure.
    #
    # Rich household types (one-person, multigenerational, multi-family, etc.)
    # are derived AFTER household construction, not enforced here.


    return constraints  # return the 0/1 mask tensor


# ---- from generate_synth_pop.ipynb cell 16 ----
# Load seed from microsample
def load_seed(df_indiv, fast):
    df_indiv = map_age_grp(df_indiv)
    # df_indiv = map_age_grp_new(df_indiv)

    df_indiv = map_hdgree(df_indiv)
    df_indiv = map_lfact(df_indiv)
    df_indiv = map_hhsize(df_indiv)
    df_indiv = map_totinc(df_indiv)
    df_indiv = map_cfstat(df_indiv)
    df_indiv = map_prihm(df_indiv)
    

    n_sex = len(df_indiv['Gender'].unique())
    n_age = len(df_indiv['agegrp'].unique())
    n_prihm = len(df_indiv['prihm'].unique())
    n_hdgree = len(df_indiv['hdgree'].unique())
    n_lfact = len(df_indiv['lfact'].unique())
    n_hhsize = len(df_indiv['hhsize'].unique())
    n_totinc = len(df_indiv['TotInc'].unique())
    n_cfstat = len(df_indiv['cfstat'].unique())

    cols = ["Gender", "prihm", 'agegrp', "hdgree", "lfact", "hhsize", "TotInc"]
    shape = [n_sex, n_prihm, n_age, n_hdgree, n_lfact, n_hhsize, n_totinc]
    if fast:
        cols = ["Gender", "prihm", 'agegrp', "hdgree", "lfact", "hhsize", "TotInc", "cfstat"]
        shape = [n_sex, n_prihm, n_age, n_hdgree, n_lfact, n_hhsize, n_totinc, n_cfstat]

    seed = unlistify(df_indiv, cols, shape, "weight")

    # Convergence problems can occur when one of the rows is zero yet the marginal total is nonzero.
    # Can get round this by adding a small number to the seed effectively allowing zero states
    #  to be occupied with a finite probability
    seed = seed.astype(float) + 1.0  # / np.sum(seed)
    if fast:
        seed = seed * get_impossible(seed)

    return seed


# ---- from generate_synth_pop.ipynb cell 19 ----
def load_census_profile(
    data_path=None,
    rest_of_path="/Census",
    starting_row_file="/98-401-X2021006_Geo_starting_row_Quebec.csv",
    census_file='/98-401-X2021006_English_CSV_data_Quebec.csv',
    province="24",
    geocode_filter=None,
    fsa_codes=None,
    cache_path=None,
    force_rebuild_cache=False,
):
    data_path = _default_data_path() if data_path is None else data_path

    def _norm_code(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        try:
            return str(int(float(s)))
        except Exception:
            return s

    cache_fp = Path(cache_path) if cache_path is not None else None
    if cache_fp is not None and cache_fp.exists() and not force_rebuild_cache:
        if cache_fp.suffix.lower() == ".parquet":
            census = pd.read_parquet(cache_fp)
        else:
            census = pd.read_csv(cache_fp, low_memory=False)
        if "geocode_norm" not in census.columns:
            census["geocode_norm"] = census["geocode"].map(_norm_code)
        return census

    start_rows = pd.read_csv(
        data_path + rest_of_path + starting_row_file,
        dtype=str,
        encoding="latin1",
    )
    mask = start_rows["Geo Code"].str[9:11] == str(province)
    start = int(start_rows.loc[mask, "Line Number"].values[0])
    end = int(start_rows.loc[mask, "Line Number"].values[-1])
    print("start and end are:", start, " - ", end)

    census = pd.read_csv(
        data_path + rest_of_path + census_file,
        skiprows=range(1, start - 1),
        nrows=end - start,
        low_memory=False,
        usecols=[
            "ALT_GEO_CODE",
            "CHARACTERISTIC_NAME",
            "CHARACTERISTIC_ID",
            "C1_COUNT_TOTAL",
            "C2_COUNT_MEN+",
            "C3_COUNT_WOMEN+",
        ],
        encoding="latin1",
    )
    census.rename(columns={'ALT_GEO_CODE': 'geocode',
                           'CHARACTERISTIC_NAME': 'variable',
                           'CHARACTERISTIC_ID': 'variableId',
                           'C1_COUNT_TOTAL': 'total',
                           'C2_COUNT_MEN+': 'totalMale',
                           'C3_COUNT_WOMEN+': 'totalFemale'}, inplace=True)
    census['variable'] = census['variable'].str.strip().astype('string')
    census["geocode_norm"] = census["geocode"].map(_norm_code)

    keep_geocodes = set()
    if geocode_filter is not None:
        keep_geocodes |= {x for x in (_norm_code(v) for v in geocode_filter) if x is not None}
    keep_geocodes.add(str(province))

    keep_fsa = set()
    if fsa_codes is not None:
        keep_fsa = {str(v).strip().upper() for v in fsa_codes if pd.notna(v) and str(v).strip() != ""}

    if keep_geocodes or keep_fsa:
        geocode_as_str = census["geocode"].astype(str).str.strip().str.upper()
        mask = census["geocode_norm"].isin(keep_geocodes)
        if keep_fsa:
            mask = mask | geocode_as_str.isin(keep_fsa)
        census = census.loc[mask].copy()

    if cache_fp is not None:
        cache_fp.parent.mkdir(parents=True, exist_ok=True)
        if cache_fp.suffix.lower() == ".parquet":
            census.to_parquet(cache_fp, index=False)
        else:
            census.to_csv(cache_fp, index=False)

    return census


# ---- from generate_synth_pop.ipynb cell 22 ----
# Load variables identifiants in census
def load_vbs_ids(census):
    # Total population id
    total_vb_id = census.loc[census["variable"] == "Population, 2021"]['variableId'].iloc[0]
    print('total_vb_id: ', total_vb_id)

    # Total age by sex id
    total_ageby_sex_vb_id = census.loc[
        census["variable"] == "Total - Age groups of the population - 100% data"]['variableId'].iloc[0]
    print('total_ageby_sex_vb_id: ',total_ageby_sex_vb_id)
    # Total households id
    total_hh_vb_id = \
        census.loc[census["variable"] == "Private dwellings occupied by usual residents"]['variableId'].iloc[0]
    print('total_hh_vb_id: ',total_hh_vb_id)
    # Total by age id
    age_vb = {}
    for i in range(0, 85, 5):
        age_vb[i] = census.loc[census["variable"] ==
                               str(i) + " to " + str(i + 4) + " years"]['variableId'].iloc[0]
    age_vb[85] = census.loc[census["variable"] == "85 years and over"]['variableId'].iloc[0]
    print('age_vb: ', age_vb)
    '''
    age_vb = {}
    age_vb[0] = 9
    age_vb[1] = 13
    age_vb[2] = 25
    age_vb[3] = 26
    age_vb[4] = 27
    age_vb[5] = 28
    age_vb[6] = 29
    '''

    # Total by hdgree id
    hdgree_vb = {}
    id_start = census.loc[census["variable"] == "Total - Highest certificate, diploma or degree for the population " \
                                                "aged 15 years and over in private households - 25% sample data"][
                   'variableId'].iloc[0] + 1
    for i in range(0, 3):
        hdgree_vb[i] = id_start + i
    print('hdgree_vb: ', hdgree_vb)

    # Total by lfact id
    lfact_vb = {}
    id_start = census.loc[census["variable"] == "Total - Labour force aged 15 years and over by class of worker including job permanency - 25% sample data"]['variableId'].iloc[0] + 2
    for i in range(0, 3):
        lfact_vb[i] = id_start + i
    print('lfact_vb: ', lfact_vb)
    
    # Total by hhsize id
    hhsize_vb = {}
    id_start = \
        census.loc[census["variable"] == "Total - Private households by household size - 100% data"]['variableId'].iloc[
            0] + 1
    for i in range(0, 5):
        hhsize_vb[i] = id_start + i
    print('hhsize_vb: ', hhsize_vb)
    # Total by totinc id
    totinc_vb = {}
    id_start = census.loc[census["variable"] ==
                          "Total - Total income groups in 2020 for the population aged 15 years and over in private households - 100% data"]['variableId'].iloc[0] + 4
    # for i in range(0, 6):
    #    totinc_vb[i] = id_start + i*2
    totinc_vb[0] = id_start
    totinc_vb[1] = id_start + 2
    totinc_vb[2] = id_start + 6
    totinc_vb[3] = id_start + 10
    print('totinc_vb: ', totinc_vb)

    # -------------------------------------------------
    # CFSTAT variable IDs (SEED-SAFE, PUMF-ALIGNED)
    # -------------------------------------------------
    # These correspond to COARSE household composition
    # that exists explicitly in PUMF and is safe for IPF.
    #
    # New CFSTAT codes:
    # 0 = Couple without children
    # 1 = Couple with children
    # 2 = One-parent family
    # 3 = Non-census-family household
    # 4 = One-person household
    # -------------------------------------------------

    cfstatVb = {}

    cfstatVb[0] = census.loc[
        census["variable"] == "Without children"
    ]['variableId'].iloc[0]  # couples without children

    cfstatVb[1] = census.loc[
        census["variable"] == "With children"
    ]['variableId'].iloc[0]  # couples with children

    cfstatVb[2] = census.loc[
        census["variable"] == "One-parent-family households"
    ]['variableId'].iloc[0]  # one-parent families

    cfstatVb[3] = census.loc[
        census["variable"] == "Two-or-more-person non-census-family households"
    ]['variableId'].iloc[0]  # roommates / relatives (non-census-family)

    cfstatVb[4] = census.loc[
        census["variable"] == "One-person households"
    ]['variableId'].iloc[0]  # one-person households

    print("cfstatVb (seed-safe):", cfstatVb)

    # -------------------------------------------------
    # Household / family averages (NOT IPF constraints)
    # -------------------------------------------------
    # These are retained ONLY for:
    # - validation
    # - post-household household-type classification
    # - computing average persons per HH type
    # -------------------------------------------------

    cfstat_sizeVb = {}

    cfstat_sizeVb["0"] = census.loc[
        census["variable"] == "Average number of children in census families with children"
    ]['variableId'].iloc[0]

    cfstat_sizeVb["1"] = census.loc[
        census["variable"] == "Average household size"
    ]['variableId'].iloc[0]

    cfstat_sizeVb["2"] = census.loc[
        census["variable"] == "Average size of census families"
    ]['variableId'].iloc[0]

    print("cfstat_sizeVb (post-processing only):", cfstat_sizeVb)


    return total_vb_id, total_ageby_sex_vb_id, total_hh_vb_id, age_vb, hdgree_vb, lfact_vb, hhsize_vb, totinc_vb, cfstatVb, cfstat_sizeVb


# ---- from generate_synth_pop.ipynb cell 28 ----
# new load_province_marginals (18 age cats: 5-year bins + 85+)
def load_province_marginals(da_census, province_census):
    total_age = {}
    total_age_f = {}
    total_age_m = {}
    total_hh_size = {}

    # Province total population (for scaling)
    population_province = int(
        province_census.loc[province_census["variableId"] == total_vb_id, "total"].iloc[0]
    )

    # DA total population
    total_pop = int(
        da_census.loc[da_census["variableId"] == total_vb_id, "total"].iloc[0]
    )

    # Sex totals (scaled from province distribution)
    total_male = int(
        total_pop
        * int(province_census.loc[province_census["variableId"] == total_age_by_sex_vb_id, "totalMale"].iloc[0])
        / population_province
    )
    total_female = total_pop - total_male

    # --- Build fine 5-year scaled counts (18 categories) ---
    # We store using age-index 0..17 (NOT the age label 0,5,10,...,85)
    # idx 0=0-4, 1=5-9, ..., 16=80-84, 17=85+
    fine_total = {}
    fine_male = {}
    fine_female = {}

    ages = list(range(0, 86, 5))  # [0,5,10,...,85] -> 18 bins

    for idx, a in enumerate(ages):
        fine_total[idx] = int(
            total_pop
            * int(province_census.loc[province_census["variableId"] == age_vb[a], "total"].iloc[0])
            / population_province
        )
        fine_male[idx] = int(
            total_pop
            * int(province_census.loc[province_census["variableId"] == age_vb[a], "totalMale"].iloc[0])
            / population_province
        )
        fine_female[idx] = fine_total[idx] - fine_male[idx]

    # --- Copy fine 18-bin values into output dicts ---
    for idx in range(len(ages)):  # 0..17
        total_age[idx] = fine_total[idx]
        total_age_m[idx] = fine_male[idx]
        total_age_f[idx] = fine_female[idx]

    # --- Household size (unchanged) ---
    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = int(
            total_pop
            * int(int(province_census.loc[province_census["variableId"] == hhsize_vb[i], "total"].iloc[0]) * (i + 1))
            / population_province
        )

    return total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size


# ---- from generate_synth_pop.ipynb cell 29 ----
def load_da_marginals(da_census):
    total_age = {}
    total_age_f = {}
    total_age_m = {}
    total_hh_size = {}

    total_pop = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['total'].iloc[0])
    print(str(total_pop) + " individuals in the DA")
    total_male = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['totalMale'].iloc[0])
    total_female = int(da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['totalFemale'].iloc[0])

    for i in range(0, 86, 5):
        # for i in range(0, len(age_vb)):
        total_age[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['total'].iloc[0])
        total_age_m[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['totalMale'].iloc[0])
        total_age_f[i] = int(da_census.loc[da_census["variableId"] == age_vb[i]]['totalFemale'].iloc[0])

    for i in range(0, len(hhsize_vb)):
        total_hh_size[i] = int(da_census.loc[da_census["variableId"] == hhsize_vb[i]]['total'].iloc[0]) * (i + 1)
    return total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size


def load_marginals_age_sex_hh(da_census, province_census):
    # if data for DA not available, use distribution of province
    total_pop_value = da_census.loc[da_census["variableId"] == total_age_by_sex_vb_id]['total'].iloc[0]
    if (total_pop_value == "x") or (total_pop_value == "F"):
        print("Census data not available for DA population, use province data")
        return load_province_marginals(da_census, province_census)
    else:
        return load_da_marginals(da_census)


def load_marginals_hdegree(da_census, province_census, total_pop):
    total_hdgree = {}
    # if data for DA not available, use distribution of province
    total_hdegree_value = da_census.loc[da_census["variableId"] == hdgree_vb[0]]['total'].iloc[0]
    if (total_hdegree_value == "x") or (total_hdegree_value == "F"):
        print("Census data not available for DA higher degree, use province data")
        for i in range(0, len(hdgree_vb)):
            total_hdgree[i] = int(total_pop * int(
                province_census.loc[province_census["variableId"] == hdgree_vb[i]]['total'].iloc[0]) / int(
                province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
    else:
        for i in range(0, len(hdgree_vb)):
            total_hdgree[i] = int(da_census.loc[da_census["variableId"] == hdgree_vb[i]]['total'].iloc[0])
    return total_hdgree


def load_marginals_lfact(da_census, province_census, total_pop):
    total_lfact = {}
    # if data for DA not available, use distribution of province
    total_lfact_value = da_census.loc[da_census["variableId"] == lfact_vb[0]]['total'].iloc[0]
    if (total_lfact_value == "x") or (total_lfact_value == "F"):
        print("Census data not available for DA employment, use province data")
        for i in range(0, len(lfact_vb)):
            total_lfact[i] = int(total_pop * int(
                province_census.loc[province_census["variableId"] == lfact_vb[i]]['total'].iloc[0]) / int(
                province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
    else:
        for i in range(0, len(lfact_vb)):
            total_lfact[i] = int(da_census.loc[da_census["variableId"] == lfact_vb[i]]['total'].iloc[0])
    return total_lfact


def load_marginals_totinc(da_census, province_census, total_pop):
    total_inc = {}
    # if data for DA not available, use distribution of province
    total_totinc_value = da_census.loc[da_census["variableId"] == totinc_vb[0]]['total'].iloc[0]
    if (total_totinc_value == "x") or (total_totinc_value == "F"):
        print("Census data not available for DA income, use province data")
        total_pop_prov = int(province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0])
        # for i in range(0, len(totinc_vb)):
        # total_inc[i] = int(total_pop * int((
        #    province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0])) / int(
        #    province_census.loc[province_census["variableId"] == total_vb_id]['total'].iloc[0]))
        total_inc[0] = int(total_pop * int(
            int(province_census.loc[province_census["variableId"] == totinc_vb[0]]['total'].iloc[0]) +
            int(province_census.loc[province_census["variableId"] == (totinc_vb[0] + 1)]['total'].iloc[
                    0])) / total_pop_prov)
        for i in range(1, 3):
            total_inc[i] = int(total_pop * int(
                int(province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 1)]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 2)]['total'].iloc[0]) +
                int(province_census.loc[province_census["variableId"] == (totinc_vb[i] + 3)]['total'].iloc[
                        0])) / total_pop_prov)
        total_inc[3] = int(total_pop * int(
            int(province_census.loc[province_census["variableId"] == totinc_vb[3]]['total'].iloc[0])) / total_pop_prov)
    else:
        # for i in range(0, len(totinc_vb)):
        # total_inc[i] = int(da_census.loc[da_census["variableId"] == totinc_vb[i]]['total'].iloc[0])
        total_inc[0] = int(da_census.loc[da_census["variableId"] == totinc_vb[0]]['total'].iloc[0]) + \
                       int(da_census.loc[da_census["variableId"] == (totinc_vb[0] + 1)]['total'].iloc[0])
        for i in range(1, 3):
            total_inc[i] = int(da_census.loc[da_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 1)]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 2)]['total'].iloc[0]) + \
                           int(da_census.loc[da_census["variableId"] == (totinc_vb[i] + 3)]['total'].iloc[0])
        total_inc[3] = int(da_census.loc[da_census["variableId"] == totinc_vb[3]]['total'].iloc[0])
    return total_inc


# ---- from generate_synth_pop.ipynb cell 30 ----
# new load_marginals_cfstat

def _get_total_or_none(census_df, variable_id):
    """Return the 'total' value for a variableId, or None if missing/suppressed."""
    if variable_id is None:
        return None
    s = census_df.loc[census_df["variableId"] == variable_id, "total"]
    if s.empty:
        return None
    v = s.iloc[0]
    if v in ("x", "F"):
        return None
    return v

def _as_float(x):
    """Convert census 'total' values to float safely."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _as_int(x):
    """Convert census 'total' values to int safely."""
    if x is None:
        return None
    try:
        return int(float(x))
    except Exception:
        return None

def _round_to_total(values_dict, target_total):
    """
    Largest-remainder rounding:
    - floor everything
    - distribute remaining persons to largest fractional parts
    Ensures sum == target_total (if possible).
    """
    keys = list(values_dict.keys())
    floats = {k: float(values_dict[k]) for k in keys}

    floors = {k: int(math.floor(floats[k])) for k in keys}
    cur_sum = sum(floors.values())
    remainder = int(target_total - cur_sum)

    if remainder == 0:
        return floors

    # fractional parts for distributing the remainder
    frac = sorted(
        [(k, floats[k] - math.floor(floats[k])) for k in keys],
        key=lambda t: t[1],
        reverse=True
    )

    # add/subtract until we match target_total
    if remainder > 0:
        for i in range(remainder):
            floors[frac[i % len(frac)][0]] += 1
    else:
        # if we need to remove people, remove from smallest fractional parts but don't go <0
        frac_rev = sorted(frac, key=lambda t: t[1])  # smallest first
        to_remove = -remainder
        i = 0
        while to_remove > 0 and i < len(frac_rev) * 10:
            k = frac_rev[i % len(frac_rev)][0]
            if floors[k] > 0:
                floors[k] -= 1
                to_remove -= 1
            i += 1

    return floors
def load_marginals_cfstat(da_census, province_census, total_pop):
    """
    Returns PERSON-count marginals for PUMF-safe CFSTAT categories (0..4).

    CFSTAT codes:
      0 = Couple without children
      1 = Couple with children
      2 = One-parent family
      3 = Non-census-family household
      4 = One-person household

    These marginals are used ONLY for IPF.
    Rich household types are derived AFTER household construction.
    """

    total_cfstat = {}

    # --- 1) Check DA availability ---
    da_check = _get_total_or_none(da_census, cfstat_vb[0])
    use_province = (da_check is None)

    if use_province:
        print("CFSTAT not available at DA level, scaling province distribution.")
        pop_prov = _as_int(_get_total_or_none(province_census, total_vb_id))
        if pop_prov is None or pop_prov <= 0:
            raise ValueError("Invalid province population.")

    # --- 2) Helper to get person counts ---
    def get_persons(i):
        if not use_province:
            return _as_int(_get_total_or_none(da_census, cfstat_vb[i]))
        prov_val = _as_int(_get_total_or_none(province_census, cfstat_vb[i]))
        if prov_val is None:
            return 0
        return int(round((total_pop * prov_val) / pop_prov))

    # --- 3) Read PERSON marginals directly ---
    # These Census variables already report persons, not households
    total_cfstat[0] = get_persons(0)  # couple without children
    total_cfstat[1] = get_persons(1)  # couple with children
    total_cfstat[2] = get_persons(2)  # one-parent family
    total_cfstat[3] = get_persons(3)  # non-census-family household
    total_cfstat[4] = get_persons(4)  # one-person household

    # --- 4) Optional: force exact population sum (minor rounding drift) ---
    diff = total_pop - sum(total_cfstat.values())
    if diff != 0:
        # push correction into the largest category (robust + neutral)
        k = max(total_cfstat, key=total_cfstat.get)
        total_cfstat[k] += diff

    return total_cfstat


# ---- from generate_synth_pop.ipynb cell 31 ----
# resto of helpers

def add_missing_hdegree(total_hdgree, province_census, total_age, total_pop):
    # add no diploma count for < 15y
    total_hdgree[0] += total_age[0] + total_age[5] + total_age[10]
    # total_hdgree[0] += total_age[0] # to use if 7 age classes

    # add missing hdegree according to province distribution
    miss = total_pop - sum(total_hdgree.values())
    for i in range(0, len(hdgree_vb)):
        total_hdgree[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == hdgree_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == hdgree_vb[0] - 1]['total'].iloc[0]))
        if total_hdgree[i] < 0:
            total_hdgree[i] = 0
    while total_pop != sum(total_hdgree.values()):
        miss = total_pop - sum(total_hdgree.values())
        random_key = random.sample(list(total_hdgree), 1)[0]
        if miss > 0 or total_hdgree[random_key] > 0:
            total_hdgree[random_key] += math.copysign(1, miss)
    return total_hdgree


def add_missing_lfact(total_lfact, province_census, total_age, total_pop):
    # add no labour force count for < 15y
    total_lfact[0] += total_age[0] + total_age[5] + total_age[10]
    # total_lfact[0] += total_age[0] # to use if 7 age classes

    # add missing labour force status following province distribution
    miss = total_pop - sum(total_lfact.values())
    for i in range(0, len(lfact_vb)):
        total_lfact[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == lfact_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == lfact_vb[0] - 2]['total'].iloc[0]))
        if total_lfact[i] < 0:
            total_lfact[i] = 0
    while total_pop != sum(total_lfact.values()):
        miss = total_pop - sum(total_lfact.values())
        random_key = random.sample(list(total_lfact), 1)[0]
        if miss > 0 or total_lfact[random_key] > 0:
            total_lfact[random_key] += math.copysign(1, miss)
    return total_lfact


def add_missing_hhsize(total_hh_size, total_pop):
    # add missing hhsize in 5+ class
    miss = total_pop - sum(total_hh_size.values())
    if (miss > 0):
        total_hh_size[len(hhsize_vb) - 1] += int(miss)

    while total_pop != sum(total_hh_size.values()):
        miss = total_pop - sum(total_hh_size.values())
        random_key = random.sample(list(total_hh_size), 1)[0]
        if (miss > 0 and random_key >= miss - 1):
            random_key = miss
        if miss > 0 or total_hh_size[random_key] > random_key + 1:
            total_hh_size[random_key] += math.copysign(random_key + 1, miss)
    return total_hh_size


def add_missing_totinc(total_inc, province_census, total_age, total_pop):
    # add <20k income count for < 15y
    total_inc[0] += total_age[0] + total_age[5] + total_age[10]
    # total_inc[0] += total_age[0] # to use if 7 age classes

    # add missing income following province distribution
    miss = total_pop - sum(total_inc.values())
    for i in range(0, len(totinc_vb)):
        total_inc[i] += int(miss * int(
            province_census.loc[province_census["variableId"] == totinc_vb[i]]['total'].iloc[0]) / int(
            province_census.loc[province_census["variableId"] == totinc_vb[0] - 2]['total'].iloc[0]))
        if total_inc[i] < 0:
            total_inc[i] = 0
    while total_pop != sum(total_inc.values()):
        miss = total_pop - sum(total_inc.values())
        random_key = random.sample(list(total_inc), 1)[0]
        if miss > 0 or total_inc[random_key] > 0:
            total_inc[random_key] += math.copysign(1, miss)
    return total_inc


def add_missing_cfstat(total_cfstat, total_pop):
    # add missing cfstat in last class
    miss = total_pop - sum(total_cfstat.values())
    if (miss > 0):
        total_cfstat[len(cfstat_vb) - 1] += int(miss)
    while total_pop != sum(total_cfstat.values()):
        miss = total_pop - sum(total_cfstat.values())
        random_key = random.sample(list(total_cfstat), 1)[0]
        if miss > 0 or total_cfstat[random_key] > 0:
            total_cfstat[random_key] += math.copysign(1, miss)
    return total_cfstat


# Increment or decrement male/female count to match total
def match_sex_count_total(total_pop, total_male, total_female):
    if total_pop != total_male + total_female:
        miss = total_pop - total_male - total_female
        total_male += int(miss / 2)
        total_female = total_pop - total_male
    return total_male, total_female


# Increment or decrement age counts to match total
def match_age_count_total(total_pop, total_male, total_female, total_age, total_age_m, total_age_f):
    for i in range(0, 86, 5):
        # for i in range(0, len(age_vb)):
        total_age[i] = total_age_m[i] + total_age_f[i]
    while total_pop != sum(total_age.values()):
        miss = total_pop - sum(total_age.values())
        random_key = random.sample(list(total_age), 1)[0]
        if miss > 0 or total_age[random_key] > 0:
            if (total_male < sum(total_age_m.values()) and miss < 0) or (
                    total_male > sum(total_age_m.values()) and miss > 0):
                if miss > 0 or total_age_m[random_key] > 0:
                    total_age_m[random_key] = total_age_m[random_key] + math.copysign(1, miss)
            elif (total_female < sum(total_age_f.values()) and miss < 0) or (
                    total_female > sum(total_age_f.values()) and miss > 0):
                if miss > 0 or total_age_f[random_key] > 0:
                    total_age_f[random_key] = total_age_f[random_key] + math.copysign(1, miss)
            for i in range(0, 86, 5):
                # for i in range(0, len(age_vb)):
                total_age[i] = total_age_m[i] + total_age_f[i]
    total_male = sum(total_age_m.values())
    total_female = sum(total_age_f.values())
    return total_age, total_age_m, total_age_f, total_male, total_female


# NOT USED
# Find best rounding threshold
# Issue: stuck in local minima
def comb_opti_integerization(p, total_pop):
    # increase threshold while error decreases
    threshold = 0.1
    previous_err = total_pop
    p["result_"] = np.around(p["result"] - threshold + 0.5)
    a = humanleague.flatten(p["result_"])[0]
    err = (abs(total_pop - len(a)))
    while (err < previous_err) | (err / total_pop * 100 > 10):
        threshold = round(threshold + 0.05, 2)
        previous_err = err
        p["result_"] = np.around(p["result"] - threshold + 0.5)
        a = humanleague.flatten(p["result_"])[0]
        err = (abs(total_pop - len(a)))

    # decrease threshold by smaller steps while error decreases
    threshold = round(threshold - 0.01, 2)
    previous_err = err
    p["result_"] = np.around(p["result"] - threshold + 0.5)
    a = humanleague.flatten(p["result_"])[0]
    err = (abs(total_pop - len(a)))
    while err < previous_err:
        threshold = round(threshold - 0.01, 2)
        previous_err = err
        p["result_"] = np.around(p["result"] - threshold + 0.5)
        a = humanleague.flatten(p["result_"])[0]
        err = (abs(total_pop - len(a)))

    threshold = round(threshold + 0.01, 2)
    print(str(previous_err / total_pop * 100) + " % error in individuals count")
    p["result"] = np.around(p["result"] - threshold + 0.5)
    return p["result"]


# ---- from generate_synth_pop.ipynb cell 32 ----
def probabilistic_sampling(p, total_pop):
    """
    Integerize a fitted contingency table by sampling total_pop individuals.

    Accepts:
      - dict with key 'result'
      - tuple where p[0] is the result array
      - numpy array directly
    Returns:
      - numpy uint8 array with exactly total_pop ones.
    """

    # --- normalize to a numpy array ---
    if isinstance(p, dict):
        result_arr = p["result"]
    elif isinstance(p, tuple):
        result_arr = p[0]
    elif isinstance(p, np.ndarray):
        result_arr = p
    else:
        raise TypeError(f"Unsupported type for p: {type(p)}")

    # --- flatten probabilities ---
    probas = np.asarray(result_arr, dtype=np.float64).ravel()
    s = probas.sum()
    if s <= 0:
        raise ValueError("Result array has zero total mass; cannot sample.")
    probas /= s

    # --- sample exactly total_pop cells (no replacement) ---
    selected = np.random.choice(len(probas), total_pop, replace=False, p=probas)

    # --- rebuild integer table ---
    result = np.zeros(result_arr.shape, dtype=np.uint8)
    result.ravel()[selected] = 1

    return result


# ---- from generate_synth_pop.ipynb cell 33 ----
def synthetise_pop_da(syn_inds, DA_code, da_census, province_census, seed, fast):
    # --- Read total households for the DA ---
    total_hh = int(
        da_census.loc[da_census["variableId"] == total_hh_vb_id, "total"].iloc[0]
    )

    # --- Load base marginals (DA first, else province fallback inside function) ---
    total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size = \
        load_marginals_age_sex_hh(da_census, province_census)
    print("age_sex_hh marginals loaded")

    total_hdgree = load_marginals_hdegree(da_census, province_census, total_pop)
    print("hdegree marginals loaded")

    total_lfact = load_marginals_lfact(da_census, province_census, total_pop)
    print("lfact marginals loaded")

    total_inc = load_marginals_totinc(da_census, province_census, total_pop)
    print("totinc marginals loaded")

    total_cfstat = load_marginals_cfstat(da_census, province_census, total_pop)
    print("cfstat marginals loaded")

    # --- Safety: households cannot exceed population ---
    total_hh = min(total_pop, total_hh)

    # --- Fill missing categories so each marginal sums to total_pop ---
    print("Add missing hdegree...")
    total_hdgree = add_missing_hdegree(total_hdgree, province_census, total_age, total_pop)

    print("Add missing lfact...")
    total_lfact = add_missing_lfact(total_lfact, province_census, total_age, total_pop)

    print("Add missing hhsize...")
    total_hh_size = add_missing_hhsize(total_hh_size, total_pop)

    print("Add missing income...")
    total_inc = add_missing_totinc(total_inc, province_census, total_age, total_pop)

    print("Add missing cfstat...")
    total_cfstat = add_missing_cfstat(total_cfstat, total_pop)

    # --- Enforce sex and age internal consistency ---
    print("Match sex counts to total...")
    total_male, total_female = match_sex_count_total(total_pop, total_male, total_female)

    print("Match age counts to total...")
    total_age, total_age_m, total_age_f, total_male, total_female = match_age_count_total(
        total_pop, total_male, total_female, total_age, total_age_m, total_age_f
    )

    # --- Build marginal arrays in the exact axis order expected by seed ---
    print("Gather marginals...")

    # sex axis: 0=female, 1=male
    marginal_sex = np.array([total_female, total_male], dtype=float)

    # prihm axis: 0=no, 1=yes
    marginal_prihm = np.array([total_pop - total_hh, total_hh], dtype=float)

    # age axis: whatever your seed uses (must match seed.shape[2])
    marginal_age = np.array(list(total_age.values()), dtype=float)

    # age_by_sex axis: 2 x n_age
    marginal_age_by_sex = np.array(
        [list(total_age_f.values()), list(total_age_m.values())],
        dtype=float
    )

    # hdgree axis
    marginal_hdgree = np.array(list(total_hdgree.values()), dtype=float)

    # lfact axis
    marginal_lfact = np.array(list(total_lfact.values()), dtype=float)

    # hhsize axis
    marginal_hh_size = np.array(list(total_hh_size.values()), dtype=float)

    # income axis
    marginal_inc = np.array(list(total_inc.values()), dtype=float)

    # cfstat axis (only used in fast/IPF branch)
    marginal_cfstat = np.array(list(total_cfstat.values()), dtype=float)

    # --- Humanleague axis selectors (these are indices of the seed axes) ---
    i0 = np.array([0])      # sex
    i1 = np.array([1])      # prihm
    i2 = np.array([2])      # agegrp
    i3 = np.array([0, 2])   # age-by-sex (sex+age)
    i4 = np.array([3])      # hdgree
    i5 = np.array([4])      # lfact
    i6 = np.array([5])      # hhsize
    i7 = np.array([6])      # totinc
    i8 = np.array([7])      # cfstat (only if seed has it)

    # --- Fit joint distribution ---
    if fast:
        print("Apply IPF (fast)")

        p = humanleague.ipf(
            seed,
            [i0, i1, i2, i3, i4, i5, i6, i7, i8],
            [marginal_sex, marginal_prihm, marginal_age, marginal_age_by_sex,
             marginal_hdgree, marginal_lfact, marginal_hh_size, marginal_inc, marginal_cfstat]
        )

        # Normalize: extract the fitted table regardless of return type (tuple vs dict)
        fitted = p["result"] if isinstance(p, dict) else p[0]

        # Integerize by probabilistic sampling (returns an integer array)
        fitted_int = probabilistic_sampling(fitted, total_pop)

        # Prepare output dataframe with cfstat column
        chunk = pd.DataFrame(
            columns=["sex", "prihm", "agegrp", "area", "hdgree", "lfact", "hhsize", "totinc", "cfstat"]
        )

        # Flatten the integerized table
        table = humanleague.flatten(fitted_int)

        # Assign columns from flattened table
        chunk["sex"] = table[0]
        chunk["prihm"] = table[1]
        chunk["agegrp"] = table[2]
        chunk["hdgree"] = table[3]
        chunk["lfact"] = table[4]
        chunk["hhsize"] = table[5]
        chunk["totinc"] = table[6]
        chunk["cfstat"] = table[7]

    else:
        print("Apply QISI (slower, no cfstat unless you include it explicitly)")

        p = humanleague.qisi(
            seed,
            [i0, i1, i2, i3, i4, i5, i6, i7],
            [marginal_sex, marginal_prihm, marginal_age, marginal_age_by_sex,
             marginal_hdgree, marginal_lfact, marginal_hh_size, marginal_inc]
        )

        fitted = p["result"] if isinstance(p, dict) else p[0]

        # QISI output is already integer-ish / constrained; flatten directly
        chunk = pd.DataFrame(
            columns=["sex", "prihm", "agegrp", "area", "hdgree", "lfact", "hhsize", "totinc"]
        )

        table = humanleague.flatten(fitted)

        chunk["sex"] = table[0]
        chunk["prihm"] = table[1]
        chunk["agegrp"] = table[2]
        chunk["hdgree"] = table[3]
        chunk["lfact"] = table[4]
        chunk["hhsize"] = table[5]
        chunk["totinc"] = table[6]

    # Add DA code
    chunk["area"] = int(DA_code)

    # Append to synthetic individuals dataframe
    syn_inds = pd.concat([syn_inds, chunk], ignore_index=True)

    return syn_inds


# ---- from generate_synth_pop.ipynb cell 50 ----
EPS = 1e-12

def _to_prob(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    if s <= 0:
        return np.zeros_like(x)
    return x / s

def tvd(p, q):
    """Total Variation Distance between probability vectors."""
    p = _to_prob(p)
    q = _to_prob(q)
    return 0.5 * np.abs(p - q).sum()

def jsd(p, q):
    """Jensen–Shannon Divergence (base-2). Robust with zeros."""
    p = _to_prob(p)
    q = _to_prob(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2((a[mask] + EPS) / (b[mask] + EPS)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def marginal_errors(syn_counts, cen_counts):
    """
    Compute MAE, MaxAE, MRE for count vectors.
    MRE uses cen_counts as denominator (with EPS).
    """
    syn = np.asarray(syn_counts, dtype=float)
    cen = np.asarray(cen_counts, dtype=float)
    L = min(len(syn), len(cen))
    syn, cen = syn[:L], cen[:L]

    ae = np.abs(syn - cen)
    re = ae / (cen + EPS)

    return {
        "MAE": float(ae.mean()),
        "MaxAE": float(ae.max()) if len(ae) else 0.0,
        "MRE": float(re.mean()),
        "MaxRE": float(re.max()) if len(re) else 0.0,
    }


# ---- from generate_synth_pop.ipynb cell 51 ----
def syn_marginal_counts_for_da(syn_inds, DA_code, col, n_cats=None):
    """
    Counts per category for syn individuals in a given DA.
    If n_cats is provided, reindex to 0..n_cats-1.
    Otherwise uses sorted observed categories.
    """
    df = syn_inds.loc[syn_inds["area"] == int(DA_code)]
    vc = df[col].value_counts()

    if n_cats is None:
        cats = sorted(vc.index.astype(int))
        return vc.reindex(cats, fill_value=0).values, cats

    counts = vc.reindex(range(n_cats), fill_value=0).values
    return counts, list(range(n_cats))


# ---- from generate_synth_pop.ipynb cell 52 ----
def census_marginal_counts_for_da(DA_code, census, province_census):
    """
    Returns:
      total_pop (int),
      dict of count-vectors for each variable
    """
    da_census = census.loc[census["geocode"] == DA_code]

    # Base marginals
    total_pop, total_male, total_female, total_age, total_age_m, total_age_f, total_hh_size = \
        load_marginals_age_sex_hh(da_census, province_census)

    # Assemble person-count marginals (vectors)
    out = {}

    out["sex"] = np.array([total_female, total_male], dtype=float)
    out["prihm"] = None  # not a census marginal here; you used total_hh to build it elsewhere
    out["agegrp"] = np.array(list(total_age.values()), dtype=float)
    out["hhsize"] = np.array(list(total_hh_size.values()), dtype=float)

    # Others
    total_hdgree = load_marginals_hdegree(da_census, province_census, total_pop)
    out["hdgree"] = np.array(list(total_hdgree.values()), dtype=float)

    total_lfact = load_marginals_lfact(da_census, province_census, total_pop)
    out["lfact"] = np.array(list(total_lfact.values()), dtype=float)

    total_inc = load_marginals_totinc(da_census, province_census, total_pop)
    out["totinc"] = np.array(list(total_inc.values()), dtype=float)

    total_cfstat = load_marginals_cfstat(da_census, province_census, total_pop)
    out["cfstat"] = np.array(list(total_cfstat.values()), dtype=float)

    return int(total_pop), out


# ---- from generate_synth_pop.ipynb cell 53 ----
def count_zero_cell_violations(syn_table, forbidden_mask):
    """
    syn_table: ndarray of integerized counts (same shape as seed)
    forbidden_mask: boolean ndarray True where state is forbidden
    returns #individuals in forbidden states and #cells violated
    """
    syn_table = np.asarray(syn_table)
    forbidden_mask = np.asarray(forbidden_mask, dtype=bool)

    violated_cells = (syn_table > 0) & forbidden_mask
    num_cells = int(violated_cells.sum())
    num_inds = int(syn_table[violated_cells].sum())
    return num_inds, num_cells


# ---- from generate_synth_pop.ipynb cell 54 ----
def quality_report_for_da(
    DA_code,
    syn_inds,
    census,
    province_census,
    var_list=("sex", "agegrp", "hdgree", "lfact", "hhsize", "totinc", "cfstat"),
    normalize_for_tvd_jsd=True,
    syn_table=None,
    seed_or_constraints=None,   # pass seed or constraints array to check violations
):
    """
    Returns:
      - summary dict
      - per-variable DataFrame with metrics
    """

    total_pop_cen, cen = census_marginal_counts_for_da(DA_code, census, province_census)

    syn_da = syn_inds.loc[syn_inds["area"] == int(DA_code)]
    total_pop_syn = int(len(syn_da))

    rows = []
    for v in var_list:
        if v not in cen or cen[v] is None:
            continue

        cen_counts = cen[v]
        n_cats = len(cen_counts)

        syn_counts, cats = syn_marginal_counts_for_da(syn_inds, DA_code, v, n_cats=n_cats)

        errs = marginal_errors(syn_counts, cen_counts)

        # distribution metrics on probabilities
        if normalize_for_tvd_jsd:
            _tvd = tvd(syn_counts, cen_counts)
            _jsd = jsd(syn_counts, cen_counts)
        else:
            _tvd = np.nan
            _jsd = np.nan

        rows.append({
            "variable": v,
            "n_cats": n_cats,
            "TVD": _tvd,
            "JSD": _jsd,
            **errs
        })

    df_metrics = pd.DataFrame(rows).sort_values("variable")

    # Summary (headline numbers)
    summary = {
        "DA": int(DA_code),
        "N_syn": total_pop_syn,
        "N_census_total": int(total_pop_cen),
        "abs_total_diff": int(abs(total_pop_syn - total_pop_cen)),
        "mean_TVD": float(df_metrics["TVD"].mean()) if len(df_metrics) else np.nan,
        "mean_JSD": float(df_metrics["JSD"].mean()) if len(df_metrics) else np.nan,
        "mean_MAE": float(df_metrics["MAE"].mean()) if len(df_metrics) else np.nan,
        "max_MaxAE": float(df_metrics["MaxAE"].max()) if len(df_metrics) else np.nan,
    }

    # Optional: zero-cell/constraint violations
    if syn_table is not None and seed_or_constraints is not None:
        forbidden = (np.asarray(seed_or_constraints) == 0)
        viol_inds, viol_cells = count_zero_cell_violations(syn_table, forbidden)
        summary["forbidden_inds"] = viol_inds
        summary["forbidden_cells"] = viol_cells

    return summary, df_metrics
