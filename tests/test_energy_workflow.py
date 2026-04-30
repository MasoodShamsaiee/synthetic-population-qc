import pandas as pd


def test_build_workflow_plan_exposes_workflow_steps():
    from synthetic_population_qc.energy_workflow import _build_workflow_plan

    support_df = pd.DataFrame(
        {
            "attribute": ["tenure", "commute_mode"],
            "unit": ["household", "person"],
            "assignment_route": ["direct_household_assignment", "sparse_fallback"],
            "support_class": ["stable", "moderately_sparse"],
        }
    )

    plan = _build_workflow_plan(
        province="24",
        geography_scope="montreal",
        data_root="C:/tmp/data",
        base_population_path=None,
        da_codes=["24660001", "24660002"],
        support_df=support_df,
    )

    assert plan["workflow_steps"][1] == "assess_support_and_plan"
    assert plan["workflow_steps"][3] == "assign_household_attributes"
    assert plan["workflow_steps"][4] == "assign_person_attributes"
    assert plan["selected_da_count"] == 2


def test_assignment_route_rows_downgrade_when_conditional_support_is_too_low():
    from synthetic_population_qc.energy_workflow import _assignment_route_rows_for_da

    support_map = {
        "citizenship_status": {
            "assignment_route": "direct_person_assignment",
            "recommended_cond_cols": ["sex", "age_group", "labour_force_status"],
            "min_conditional_weight": 25.0,
        }
    }
    sparse_seed = pd.DataFrame(
        {
            "weight": [10.0, 5.0],
            "sex": ["female", "male"],
            "age_group": [1, 2],
            "labour_force_status": [1867, 1867],
            "citizenship_status": ["canadian_citizen", "not_canadian_citizen"],
        }
    )

    rows = _assignment_route_rows_for_da(
        da_code="24660001",
        unit="person",
        attrs=["citizenship_status"],
        support_map=support_map,
        seed_df=sparse_seed,
    )

    assert rows[0]["planned_route"] == "direct_person_assignment"
    assert rows[0]["selected_route"] == "sparse_fallback"
    assert rows[0]["downgraded_to_sparse"] is True


def test_household_coherence_audit_flags_invalid_size_type_pairs():
    from synthetic_population_qc.energy_workflow import _build_household_coherence_audit

    households = pd.DataFrame(
        {
            "area": ["1", "1", "1", "1"],
            "household_id": ["a", "b", "c", "d"],
            "household_size": ["3", "1", "2", "2"],
            "household_type": [
                "couple_without_children",
                "one_person",
                "couple_with_children",
                "one_parent",
            ],
        }
    )

    audit = _build_household_coherence_audit(households)

    assert "couple_without_children_requires_size_2" in audit["coherence_issue"].tolist()
    assert "couple_with_children_requires_size_3plus" in audit["coherence_issue"].tolist()
    assert len(audit) == 2


def test_map_base_people_to_core_schema_preserves_existing_household_size_labels():
    from synthetic_population_qc.energy_workflow import _map_base_people_to_core_schema

    base_people = pd.DataFrame(
        {
            "sex": [1],
            "agegrp": [3],
            "lfact": [1867],
            "hhsize": [2],
            "hdgree": [1],
            "totinc": [1],
            "cfstat": [1],
            "hhtype": [0],
        }
    )
    core_levels = {
        "age_group": [1, 2, 3, 4],
        "labour_force_status": [1867],
        "education_level": [1],
        "household_income": [1],
        "family_status": [1],
    }

    out = _map_base_people_to_core_schema(base_people, core_levels)

    assert out.loc[0, "household_size"] == "2"
    assert out.loc[0, "household_type"] == "couple_without_children"


def test_summarize_aligned_outputs_includes_core_and_extension_attributes():
    from synthetic_population_qc.energy_workflow import _summarize_aligned_outputs

    people = pd.DataFrame(
        {
            "area": ["24660001", "24660001", "24660001"],
            "sex": ["female", "male", "female"],
            "age_group": [10, 11, 10],
            "education_level": [1684, 1685, 1684],
            "labour_force_status": [1867, 1867, 1869],
            "household_size": ["1", "2", "1"],
            "household_income": [695, 697, 695],
            "family_status": [1, 2, 1],
            "citizenship_status": ["canadian_citizen", "not_canadian_citizen", "canadian_citizen"],
            "immigrant_status": ["non_immigrant", "immigrant", "non_immigrant"],
            "commute_mode": ["private_vehicle", "public_transit", pd.NA],
            "commute_duration": ["lt_15_min", "15_to_29_min", pd.NA],
        }
    )
    households = pd.DataFrame(
        {
            "area": ["24660001", "24660001"],
            "household_id": ["h1", "h2"],
            "household_size": ["1", "2"],
            "household_type": ["one_person", "couple_without_children"],
            "dwelling_type": ["apartment", "single_detached_house"],
            "tenure": ["owner", "renter_or_band"],
            "bedrooms": ["1_bedroom", "2_bedrooms"],
            "period_built": ["1991_to_2000", "2001_to_2005"],
            "dwelling_condition": ["regular_maintenance", "major_repairs"],
            "core_housing_need": ["not_in_core_need", "in_core_need"],
        }
    )
    context_tables = {
        "immigration_citizenship": pd.DataFrame(
            {
                "da_code": ["24660001"],
                "citizen_col": [2],
                "noncitizen_col": [1],
                "nonimm_col": [2],
                "imm_col": [1],
                "npr_col": [0],
            }
        ),
        "commute": pd.DataFrame(
            {
                "da_code": ["24660001"],
                "car_col": [1],
                "transit_col": [1],
                "walk_col": [0],
                "bike_col": [0],
                "other_col": [0],
                "lt15_col": [1],
                "m15_29_col": [1],
                "m30_44_col": [0],
                "m45_59_col": [0],
                "m60_col": [0],
            }
        ),
        "housing": pd.DataFrame(
            {
                "da_code": ["24660001"],
                "owner_col": [1],
                "renter_col": [1],
                "bed1_col": [1],
                "bed2_col": [1],
                "pb_1991_2000_col": [1],
                "pb_2001_2005_col": [1],
                "maint_col": [1],
                "major_col": [1],
                "need_no_col": [1],
                "need_yes_col": [1],
            }
        ),
        "dwelling_characteristics": pd.DataFrame(
            {
                "da_code": ["24660001"],
                "apt_col": [1],
                "sdh_col": [1],
            }
        ),
    }

    from synthetic_population_qc import enrichment as enr

    enr.PERSON_ATTR_LABELS["citizenship_status"] = {
        "canadian_citizen": "citizen_col",
        "not_canadian_citizen": "noncitizen_col",
    }
    enr.PERSON_ATTR_LABELS["immigrant_status"] = {
        "non_immigrant": "nonimm_col",
        "immigrant": "imm_col",
        "non_permanent_resident": "npr_col",
    }
    enr.PERSON_ATTR_LABELS["commute_mode"] = {
        "car_truck_van": "car_col",
        "public_transit": "transit_col",
        "walked": "walk_col",
        "bicycle": "bike_col",
        "other_method": "other_col",
    }
    enr.PERSON_ATTR_LABELS["commute_duration"] = {
        "lt_15_min": "lt15_col",
        "15_to_29_min": "m15_29_col",
        "30_to_44_min": "m30_44_col",
        "45_to_59_min": "m45_59_col",
        "60plus_min": "m60_col",
    }
    enr.HOUSEHOLD_ATTR_LABELS["tenure"] = {"owner": "owner_col", "renter": "renter_col"}
    enr.HOUSEHOLD_ATTR_LABELS["bedrooms"] = {"1_bedroom": "bed1_col", "2_bedrooms": "bed2_col"}
    enr.HOUSEHOLD_ATTR_LABELS["period_built"] = {
        "1991_to_2000": "pb_1991_2000_col",
        "2001_to_2005": "pb_2001_2005_col",
    }
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_condition"] = {
        "regular_or_minor_repairs": "maint_col",
        "major_repairs_needed": "major_col",
    }
    enr.HOUSEHOLD_ATTR_LABELS["core_housing_need"] = {
        "not_in_core_need": "need_no_col",
        "in_core_need": "need_yes_col",
    }
    enr.HOUSEHOLD_ATTR_LABELS["dwelling_type"] = {
        "duplex_apartment": "apt_col",
        "single_detached": "sdh_col",
    }

    summary, details = _summarize_aligned_outputs(
        people,
        households,
        context_tables,
        reference_people_df=people.copy(),
        reference_households_df=households.copy(),
    )

    by_attr = summary.set_index("attribute")
    assert "sex" in by_attr.index
    assert "household_type" in by_attr.index
    assert "citizenship_status" in by_attr.index
    assert "household_size" in by_attr.index
    assert by_attr.loc["sex", "reference_source"] == "base_population"
    assert by_attr.loc["citizenship_status", "reference_source"] == "direct_da"
    assert by_attr.loc["sex", "workflow_role"] == "core_generation"
    assert pd.notna(by_attr.loc["citizenship_status", "tvd"])
    assert not details.empty
    detail_lookup = details.set_index(["attribute", "category"])
    assert ("age_group", "0_to_4") in detail_lookup.index
    assert ("labour_force_status", "employed") in detail_lookup.index
    assert ("family_status", "couple_with_children_family_member") in detail_lookup.index


def test_people_output_uses_supported_columns_without_alias_duplicates():
    from synthetic_population_qc.energy_workflow import _canonicalize_people_output

    people = pd.DataFrame(
        {
            "area": ["1"],
            "HID": ["10"],
            "sex": [1],
            "agegrp": [3],
            "hdgree": [2],
            "lfact": [1867],
            "hhsize": [2],
            "totinc": [1],
            "cfstat": [1],
            "hhtype": [0],
            "age_group": [11],
            "education_level": [1685],
            "labour_force_status": [1867],
            "household_size": ["2"],
            "household_income": [697],
            "family_status": [1],
            "household_type": ["couple_without_children"],
            "person_uid": [0],
            "citizenship_status": ["canadian_citizen"],
            "immigrant_status": ["non_immigrant"],
            "commute_mode": ["public_transit"],
            "commute_duration": ["15_to_29_min"],
        }
    )

    out = _canonicalize_people_output(people)

    assert "sex" in out.columns
    assert "agegrp" not in out.columns
    assert "hhsize" not in out.columns
    assert "hhsize_core" not in out.columns
    assert "hhtype" not in out.columns
    assert "sex" in out.columns
    assert "age_group" in out.columns
    assert "education_level" in out.columns
    assert "labour_force_status" in out.columns
    assert "household_income" in out.columns
    assert "family_status" in out.columns
    assert "household_size" in out.columns
    assert "household_type" in out.columns
    assert out.loc[0, "age_group"] == "5_to_14"
    assert out.loc[0, "education_level"] == "high_school_or_equivalent"
    assert out.loc[0, "labour_force_status"] == "employed"
    assert out.loc[0, "household_income"] == "20k_to_59k"
    assert out.loc[0, "family_status"] == "couple_with_children_family_member"


def test_people_output_supports_coarser_age_scheme():
    from synthetic_population_qc.energy_workflow import _canonicalize_people_output

    people = pd.DataFrame(
        {
            "area": ["1", "1"],
            "HID": ["10", "11"],
            "sex": ["male", "female"],
            "age_group": [10, 11],
            "education_level": [1685, 1685],
            "labour_force_status": [1867, 1867],
            "household_income": [697, 697],
            "family_status": [1, 1],
            "household_size": ["2", "2"],
            "household_type": ["couple_without_children", "couple_without_children"],
            "person_uid": [0, 1],
        }
    )

    out = _canonicalize_people_output(people, age_group_scheme="coarse_10")

    assert sorted(out["age_group"].tolist()) == ["0_to_14", "0_to_14"]
