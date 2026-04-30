import numpy as np
import pandas as pd


def test_build_joint_workflow_plan_artifacts_exposes_unified_controls():
    from synthetic_population_qc.joint_fit import DERIVED_OUTPUT_ATTRIBUTES, build_joint_workflow_plan_artifacts

    support_df, workflow_plan = build_joint_workflow_plan_artifacts(
        context_tables={},
        da_codes=["24660001", "24660002"],
    )

    assert workflow_plan["workflow_method"] == "joint_ipu_v1"
    assert workflow_plan["selected_da_count"] == 2
    assert "fit_joint_household_donors" in workflow_plan["workflow_steps"]
    assert (support_df["workflow_role"] == "joint_fit").all()
    assert (support_df["assignment_route"] == "joint_fit").all()
    assert "control_tier" in support_df.columns
    assert workflow_plan["derived_output_attributes"] == list(DERIVED_OUTPUT_ATTRIBUTES)
    hh_type_row = support_df.loc[support_df["attribute"] == "household_type"].iloc[0]
    assert hh_type_row["target_source"] == "household_type_size_detailed"
    sex_row = support_df.loc[support_df["attribute"] == "sex"].iloc[0]
    assert sex_row["target_source"] == "age_sex_core"
    assert "family_status" not in support_df["attribute"].tolist()


def test_family_status_is_derived_not_jointly_controlled():
    from synthetic_population_qc.joint_fit import DERIVED_OUTPUT_ATTRIBUTES, JOINT_CONTROL_SPECS

    assert "family_status" in DERIVED_OUTPUT_ATTRIBUTES
    assert "family_status" not in [item.attribute for item in JOINT_CONTROL_SPECS]


def test_joint_weight_solver_reduces_objective():
    from synthetic_population_qc.joint_fit import JOINT_CONTROL_SPECS, _solve_joint_weights

    spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "household_type")
    household_donors = pd.DataFrame({"weight": [1.0, 1.0]})
    member_donors = pd.DataFrame()
    target_bundle = [
        {
            "spec": spec,
            "matrix": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            "target_vector": np.array([3.0, 1.0], dtype=float),
            "effective_counts": {"couple_with_children": 3, "couple_without_children": 1},
            "tier": spec.tier,
            "tier_weight": spec.tier_weight,
            "target_provenance": "direct_da",
            "smoothing_mode": "direct",
            "raw_target_total": 4,
            "effective_target_total": 4,
            "donor_support_mass": 2.0,
            "fit_error": np.nan,
            "area": "24660001",
        }
    ]

    _, history = _solve_joint_weights(
        household_donors,
        member_donors,
        target_bundle,
        household_total=4,
        max_iterations=8,
    )

    assert history
    assert history[-1]["objective"] <= history[0]["objective"]


def test_joint_summary_reports_smoothed_direct_da_for_all_attributes():
    from synthetic_population_qc.joint_fit import JOINT_CONTROL_SPECS, _summarize_joint_fit_outputs

    sex_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "sex")
    tenure_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "tenure")
    people = pd.DataFrame(
        {
            "area": ["24660001", "24660001"],
            "sex": ["female", "male"],
            "age_group": [10, 14],
            "education_level": [1684, 1685],
            "labour_force_status": [1869, 1867],
            "household_income": [695, 697],
            "family_status": [4, 0],
            "household_size": ["1", "2"],
            "household_type": ["one_person", "couple_without_children"],
            "citizenship_status": ["canadian_citizen", "canadian_citizen"],
            "immigrant_status": ["non_immigrant", "non_immigrant"],
            "commute_mode": [pd.NA, "public_transit"],
            "commute_duration": [pd.NA, "15_to_29_min"],
        }
    )
    households = pd.DataFrame(
        {
            "area": ["24660001", "24660001"],
            "household_id": ["1", "2"],
            "household_size": ["1", "2"],
            "household_type": ["one_person", "couple_without_children"],
            "tenure": ["owner", "renter_or_band"],
        }
    )
    target_bundle = [
        {
            "spec": sex_spec,
            "effective_counts": {"female": 1, "male": 1},
            "tier": sex_spec.tier,
            "tier_weight": sex_spec.tier_weight,
            "target_provenance": "direct_da",
            "smoothing_mode": "direct",
            "raw_target_total": 2,
            "effective_target_total": 2,
            "donor_support_mass": 2.0,
            "fit_error": 0.0,
        },
        {
            "spec": tenure_spec,
            "effective_counts": {"owner": 1, "renter_or_band": 1},
            "tier": tenure_spec.tier,
            "tier_weight": tenure_spec.tier_weight,
            "target_provenance": "direct_da",
            "smoothing_mode": "direct",
            "raw_target_total": 2,
            "effective_target_total": 2,
            "donor_support_mass": 2.0,
            "fit_error": 0.0,
        },
    ]

    summary, details = _summarize_joint_fit_outputs(
        people_df=people,
        households_df=households,
        target_bundle=target_bundle,
        age_group_scheme="default_15",
        age_group_breaks=None,
    )

    assert (summary["reference_source"] == "smoothed_direct_da").all()
    assert (summary["census_source"] == "smoothed_direct_da").all()
    assert (summary["workflow_role"] == "joint_fit").all()
    assert "target_provenance" in summary.columns
    assert not details.empty


def test_extract_direct_counts_splits_dwelling_condition_proxy():
    from synthetic_population_qc.joint_fit import JOINT_CONTROL_SPECS, _extract_direct_counts
    from synthetic_population_qc.enrichment import HOUSEHOLD_ATTR_LABELS

    spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "dwelling_condition")
    row = pd.Series(
        {
            HOUSEHOLD_ATTR_LABELS["dwelling_condition"]["regular_or_minor_repairs"]: 10,
            HOUSEHOLD_ATTR_LABELS["dwelling_condition"]["major_repairs_needed"]: 4,
        }
    )

    counts = _extract_direct_counts(spec, row)

    assert counts["regular_maintenance"] == 5
    assert counts["minor_repairs"] == 5
    assert counts["major_repairs"] == 4


def test_extract_direct_counts_reads_age_sex_core_targets():
    from synthetic_population_qc.joint_fit import JOINT_CONTROL_SPECS, _extract_direct_counts

    sex_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "sex")
    age_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "age_group")
    row = pd.Series(
        {
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 0 to 4 years ; Males": 5,
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 5 to 9 years ; Males": 6,
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 0 to 14 years ; Males / 10 to 14 years ; Males": 7,
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 15 to 19 years ; Males": 8,
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 15 to 64 years ; Males / 20 to 24 years ; Males": 9,
            "Age & Sex - Males / Total - Age groups of the population - 100% data ; Males / 65 years and over ; Males / 85 years and over ; Males": 12,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 0 to 4 years ; Females": 4,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 5 to 9 years ; Females": 3,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 0 to 14 years ; Females / 10 to 14 years ; Females": 2,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 15 to 19 years ; Females": 1,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 15 to 64 years ; Females / 20 to 24 years ; Females": 5,
            "Age & Sex - Females / Total - Age groups of the population - 100% data ; Females / 65 years and over ; Females / 85 years and over ; Females": 8,
        }
    )

    sex_counts = _extract_direct_counts(sex_spec, row)
    age_counts = _extract_direct_counts(age_spec, row)

    assert sex_counts == {"female": 23, "male": 47}
    assert age_counts[10] == 9
    assert age_counts[11] == 18
    assert age_counts[12] == 23
    assert age_counts[26] == 20


def test_extract_direct_counts_builds_household_size_proxy():
    from synthetic_population_qc.joint_fit import JOINT_CONTROL_SPECS, _extract_direct_counts

    size_spec = next(item for item in JOINT_CONTROL_SPECS if item.attribute == "household_size")
    row = pd.Series(
        {
            "Households by type / Total - Household type - 100% data / One-person households": 10,
            "Households by type / Total - Household type - 100% data / Two-or-more-person non-census-family households": 8,
            "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / Couple-family households / With children": 4,
            "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / Couple-family households / Without children": 3,
            "Households by type / Total - Household type - 100% data / One-census-family households without additional persons / One-parent-family households": 2,
            "Family characteristics / Total - Census families in private households by family size - 100% data / 2 persons": 5,
            "Family characteristics / Total - Census families in private households by family size - 100% data / 3 persons": 4,
            "Family characteristics / Total - Census families in private households by family size - 100% data / 4 persons": 3,
            "Family characteristics / Total - Census families in private households by family size - 100% data / 5 or more persons": 2,
        }
    )
    household_donors = pd.DataFrame(
        {
            "household_id": ["o2", "o3", "o4", "o5", "cw", "cwo", "op", "p1"],
            "household_type": ["other", "other", "other", "other", "couple_with_children", "couple_without_children", "one_parent", "one_person"],
            "household_size": ["2", "3", "4", "5plus", "4", "2", "3", "1"],
        }
    )

    size_counts = _extract_direct_counts(size_spec, row, household_donors=household_donors)

    assert sum(size_counts.values()) == 32
    assert size_counts["1"] == 10
