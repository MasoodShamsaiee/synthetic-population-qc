import pandas as pd


def test_support_assessment_marks_stable_and_sparse_attributes():
    from synthetic_population_qc.support_assessment import build_support_assessment

    person_seed = pd.DataFrame(
        {
            "weight": [60.0, 55.0, 52.0, 51.0, 15.0, 12.0],
            "sex": ["female", "female", "male", "male", "female", "male"],
            "age_group": [1, 2, 1, 2, 1, 2],
            "labour_force_status": [1867, 1867, 1867, 1867, 1867, 1867],
            "citizenship_status": ["canadian_citizen", "canadian_citizen", "not_canadian_citizen", "canadian_citizen", "canadian_citizen", "not_canadian_citizen"],
            "immigrant_status": ["non_immigrant", "immigrant", "non_immigrant", "immigrant", "non_immigrant", "immigrant"],
            "commute_mode_group": ["car_truck_van", "public_transit", "car_truck_van", "walked", "other_method", "walked"],
            "commute_duration": ["lt_15_min", "15_to_29_min", "30_to_44_min", "45_to_59_min", "60plus_min", "15_to_29_min"],
        }
    )
    household_seed = pd.DataFrame(
        {
            "weight": [80.0, 75.0, 70.0, 65.0],
            "household_size": ["1", "2", "3", "4"],
            "household_type": ["one_person", "couple_without_children", "couple_with_children", "one_parent"],
            "dwelling_type": ["apartment", "single_detached_house", "apartment", "other_dwelling"],
            "tenure": ["owner", "owner", "renter_or_band", "renter_or_band"],
            "bedrooms": ["1_bedroom", "3_bedrooms", "2_bedrooms", "4plus_bedrooms"],
            "period_built": ["1991_to_2000", "2001_to_2005", "2006_to_2010", "2011_to_2015"],
            "dwelling_condition": ["regular_maintenance", "regular_maintenance", "major_repairs", "minor_repairs"],
            "core_housing_need": ["not_in_core_need", "not_in_core_need", "in_core_need", "in_core_need"],
        }
    )
    context_tables = {
        "immigration_citizenship": pd.DataFrame(
            {
                "da_code": ["1"],
                "citizen_col_a": [100],
                "immigrant_col_a": [100],
            }
        ),
        "commute": pd.DataFrame({"da_code": ["1"], "commute_col_a": [100]}),
        "housing": pd.DataFrame({"da_code": ["1"], "housing_col_a": [100]}),
        "dwelling_characteristics": pd.DataFrame({"da_code": ["1"], "dwelling_col_a": [100]}),
    }

    from synthetic_population_qc import enrichment as enr

    enr.PERSON_ATTR_LABELS["citizenship_status"] = {"canadian_citizen": "citizen_col_a"}
    enr.PERSON_ATTR_LABELS["immigrant_status"] = {"immigrant": "immigrant_col_a"}
    enr.PERSON_ATTR_LABELS["commute_mode"] = {"private_vehicle": "commute_col_a"}
    enr.PERSON_ATTR_LABELS["commute_duration"] = {"lt_15_min": "commute_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["tenure"] = {"owner": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["bedrooms"] = {"1_bedroom": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["period_built"] = {"1991_to_2000": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_condition"] = {"regular_maintenance": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["core_housing_need"] = {"not_in_core_need": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_type"] = {"apartment": "dwelling_col_a"}

    out = build_support_assessment(
        person_seed_df=person_seed,
        household_seed_df=household_seed,
        context_tables=context_tables,
        da_codes=["1"],
    )

    by_attr = out.set_index("attribute")
    assert by_attr.loc["citizenship_status", "support_class"] == "stable"
    assert by_attr.loc["dwelling_type", "assignment_route"] == "direct_household_assignment"
    assert by_attr.loc["commute_mode", "assignment_route"] == "sparse_fallback"


def test_assign_attribute_with_fallback_uses_coarser_conditioning_when_needed():
    from synthetic_population_qc.sparse_handling import assign_attribute_with_fallback

    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "sex": ["female", "female", "male", "male"],
            "age_group": [1, 2, 1, 2],
        }
    )
    seed = pd.DataFrame(
        {
            "weight": [20.0, 5.0, 20.0, 5.0],
            "sex": ["female", "female", "male", "male"],
            "age_group": [1, 2, 1, 2],
            "attr": ["x", "x", "y", "y"],
        }
    )

    assigned, report = assign_attribute_with_fallback(
        df,
        seed_df=seed,
        attr="attr",
        target_counts={"x": 2, "y": 2},
        row_id_col="row_id",
        fallback_ladder=[["sex", "age_group"], ["sex"], []],
        min_conditional_weight=10.0,
    )

    assert assigned.value_counts().to_dict() == {"x": 2, "y": 2}
    assert report["fallback_rank"] >= 1


def test_support_assessment_includes_core_generation_attributes_without_sparse_rerouting():
    from synthetic_population_qc.support_assessment import build_support_assessment

    person_seed = pd.DataFrame(
        {
            "weight": [60.0, 55.0, 52.0, 51.0],
            "sex": ["female", "female", "male", "male"],
            "age_group": [1, 2, 1, 2],
            "education_level": [1, 2, 1, 2],
            "labour_force_status": [1867, 1867, 1867, 1867],
            "household_size": ["1", "2", "1", "2"],
            "household_income": [1, 2, 1, 2],
            "family_status": [1, 2, 1, 2],
            "citizenship_status": ["canadian_citizen", "canadian_citizen", "not_canadian_citizen", "canadian_citizen"],
            "immigrant_status": ["non_immigrant", "immigrant", "non_immigrant", "immigrant"],
            "commute_mode_group": ["car_truck_van", "public_transit", "car_truck_van", "walked"],
            "commute_duration": ["lt_15_min", "15_to_29_min", "30_to_44_min", "45_to_59_min"],
        }
    )
    household_seed = pd.DataFrame(
        {
            "weight": [80.0, 75.0, 70.0, 65.0],
            "household_size": ["1", "2", "3", "4"],
            "household_type": ["one_person", "couple_without_children", "couple_with_children", "one_parent"],
            "dwelling_type": ["apartment", "single_detached_house", "apartment", "other_dwelling"],
            "tenure": ["owner", "owner", "renter_or_band", "renter_or_band"],
            "bedrooms": ["1_bedroom", "3_bedrooms", "2_bedrooms", "4plus_bedrooms"],
            "period_built": ["1991_to_2000", "2001_to_2005", "2006_to_2010", "2011_to_2015"],
            "dwelling_condition": ["regular_maintenance", "regular_maintenance", "major_repairs", "minor_repairs"],
            "core_housing_need": ["not_in_core_need", "not_in_core_need", "in_core_need", "in_core_need"],
        }
    )
    context_tables = {
        "immigration_citizenship": pd.DataFrame({"da_code": ["1"], "citizen_col_a": [100], "immigrant_col_a": [100]}),
        "commute": pd.DataFrame({"da_code": ["1"], "commute_col_a": [100]}),
        "housing": pd.DataFrame({"da_code": ["1"], "housing_col_a": [100]}),
        "dwelling_characteristics": pd.DataFrame({"da_code": ["1"], "dwelling_col_a": [100]}),
    }

    from synthetic_population_qc import enrichment as enr

    enr.PERSON_ATTR_LABELS["citizenship_status"] = {"canadian_citizen": "citizen_col_a"}
    enr.PERSON_ATTR_LABELS["immigrant_status"] = {"immigrant": "immigrant_col_a"}
    enr.PERSON_ATTR_LABELS["commute_mode"] = {"private_vehicle": "commute_col_a"}
    enr.PERSON_ATTR_LABELS["commute_duration"] = {"lt_15_min": "commute_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["tenure"] = {"owner": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["bedrooms"] = {"1_bedroom": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["period_built"] = {"1991_to_2000": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_condition"] = {"regular_maintenance": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["core_housing_need"] = {"not_in_core_need": "housing_col_a"}
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_type"] = {"apartment": "dwelling_col_a"}

    out = build_support_assessment(
        person_seed_df=person_seed,
        household_seed_df=household_seed,
        context_tables=context_tables,
        da_codes=["1"],
    )

    by_attr = out.set_index("attribute")
    assert "sex" in by_attr.index
    assert "household_size" in by_attr.index
    assert by_attr.loc["sex", "assignment_route"] == "core_generation"
    assert by_attr.loc["household_size", "assignment_route"] == "core_generation"
    assert pd.isna(by_attr.loc["sex", "da_marginal_available"])
