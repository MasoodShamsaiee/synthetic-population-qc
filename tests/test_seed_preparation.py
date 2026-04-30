"""Regression tests for harmonized seed-preparation helpers."""

import pandas as pd


def test_prepare_person_seed_maps_new_attributes():
    from synthetic_population_qc.seed_preparation import prepare_person_seed

    df = pd.DataFrame(
        {
            "PPSORT": [1, 2],
            "AGEGRP": [10, 15],
            "Gender": [1, 2],
            "HDGREE": [2, 9],
            "LFACT": [1, 13],
            "TotInc": [25000, 90000],
            "HHSIZE": [2, 4],
            "CFSTAT": [1, 6],
            "PRIHM": [1, 0],
            "Citizen": [1, 3],
            "IMMSTAT": [1, 2],
            "MODE": [6, 2],
            "PWDUR": [2, 5],
            "DTYPE": [2, 1],
            "Tenur": [2, 1],
            "BedRm": [2, 4],
            "REPAIR": [3, 1],
            "HCORENEED_IND": [100, 0],
            "PR": ["24", "24"],
            "WEIGHT": [1.2, 0.8],
        }
    )

    out = prepare_person_seed(df)

    assert "citizenship_status" in out.columns
    assert out.loc[0, "commute_mode"] == "public_transit"
    assert out.loc[1, "tenure_proxy"] == "owner"


def test_prepare_household_and_chs_seeds():
    from synthetic_population_qc.seed_preparation import (
        prepare_chs_household_seed,
        prepare_hierarchical_household_seed,
    )

    hier = pd.DataFrame(
        {
            "HH_ID": [10, 10, 10, 11],
            "PP_ID": [101, 102, 103, 201],
            "AGEGRP": [10, 10, 5, 15],
            "GENDER": [1, 2, 2, 1],
            "HDGREE": [7, 7, 99, 2],
            "LFACT": [1, 1, 99, 13],
            "TOTINC": [50000, 42000, 0, 30000],
            "CFSTAT": [2, 2, 4, 1],
            "PRIHM": [1, 0, 0, 1],
            "HHMAINP": [1, 0, 0, 1],
            "CITIZEN": [1, 1, 1, 3],
            "IMMSTAT": [1, 1, 1, 2],
            "MODE": [2, 2, 9, 6],
            "PWDUR": [3, 3, 9, 2],
            "DTYPE": [2, 2, 2, 1],
            "TENUR": [2, 2, 2, 1],
            "BEDRM": [2, 2, 2, 4],
            "BUILT": [8, 8, 8, 4],
            "REPAIR": [2, 2, 2, 1],
            "HCORENEED_IND": [100, 100, 100, 0],
            "PR": ["24", "24", "24", "24"],
            "WEIGHT": [1.0, 1.0, 1.0, 1.0],
        }
    )
    chs = pd.DataFrame(
        {
            "PUMFID": [1],
            "PHHSIZE": [3],
            "PHHTTINC": [65000],
            "PDWLTYPE": ["6"],
            "PDV_SUIT": [1],
            "P1DCT_20": [3],
            "PRSPIMST": [2],
            "PRSPGNDR": [2],
            "POWN_20": [1],
            "PFWEIGHT": [2.5],
        }
    )

    hier_out = prepare_hierarchical_household_seed(hier)
    chs_out = prepare_chs_household_seed(chs)

    assert "period_built" in hier_out.columns
    assert hier_out.loc[hier_out["household_id"] == "10", "household_type"].iloc[0] == "couple_with_children"
    assert chs_out.loc[0, "housing_suitability"] == "suitable"


def test_derive_household_type_treats_additional_persons_as_other():
    from synthetic_population_qc.seed_preparation import _derive_household_type_from_members

    hier = pd.DataFrame(
        {
            "HH_ID": [20, 20, 20, 21, 21],
            "CFSTAT": [1, 1, 6, 1, 1],
        }
    )

    out = _derive_household_type_from_members(hier).set_index("HH_ID")

    assert out.loc[20, "household_type"] == "other"
    assert out.loc[21, "household_type"] == "couple_without_children"
