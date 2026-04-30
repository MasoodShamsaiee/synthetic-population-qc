"""Generate auditable parameter/value mapping tables for the workflow."""

from __future__ import annotations

import pandas as pd

from synthetic_population_qc.public_schema import (
    AGE_GROUP_BOUNDS,
    PUBLIC_ATTR_RENAMES,
    PUBLIC_VALUE_LABELS,
    age_group_value_labels,
)


def build_mapping_audit_df(
    *,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Export an auditable table of source-to-semantic workflow mappings."""
    rows: list[dict[str, object]] = []

    for source_name, public_name in sorted(PUBLIC_ATTR_RENAMES.items()):
        rows.append(
            {
                "section": "column_rename",
                "parameter": public_name,
                "source_name": source_name,
                "source_value": None,
                "internal_name": source_name,
                "internal_value": None,
                "public_name": public_name,
                "public_value": None,
                "action_type": "rename",
                "notes": "Canonical public output name.",
            }
        )

    age_labels = age_group_value_labels(scheme=age_group_scheme, custom_breaks=age_group_breaks)
    for code, bounds in sorted(AGE_GROUP_BOUNDS.items()):
        rows.append(
            {
                "section": "value_mapping",
                "parameter": "age_group",
                "source_name": "AGEGRP",
                "source_value": code,
                "internal_name": "age_group",
                "internal_value": code,
                "public_name": "age_group",
                "public_value": age_labels[code],
                "action_type": "label_merge",
                "notes": f"Existing harmonized age bucket bounds: {bounds[0]} to {bounds[1] if bounds[1] is not None else 'plus'}",
            }
        )

    for public_name, mapping in sorted(PUBLIC_VALUE_LABELS.items()):
        for internal_value, public_value in sorted(mapping.items(), key=lambda item: str(item[0])):
            rows.append(
                {
                    "section": "value_mapping",
                    "parameter": public_name,
                "source_name": public_name,
                "source_value": internal_value,
                "internal_name": public_name,
                    "internal_value": internal_value,
                    "public_name": public_name,
                    "public_value": public_value,
                    "action_type": "label",
                    "notes": "Harmonized internal code mapped to public category label.",
                }
            )

    seed_transform_rows = [
        ("age_group", "AGEGRP", 88, None, "drop", "Dropped as unsupported/special age code in current harmonization."),
        ("age_group", "AGEGRP", "27-29", 26, "collapse", "Detailed 90+ tail codes now collapse into one harmonized 85_plus bucket."),
        ("education_level", "HDGREE", 88, 1684, "collapse", "Treated as lowest education bucket in current harmonization."),
        ("education_level", "HDGREE", 99, 1684, "collapse", "Treated as lowest education bucket in current harmonization."),
        ("labour_force_status", "LFACT", "1,2", 1867, "collapse", "Collapsed to employed."),
        ("labour_force_status", "LFACT", "3-10", 1868, "collapse", "Collapsed to unemployed."),
        ("labour_force_status", "LFACT", "11-99", 1869, "collapse", "Collapsed to not in labour force."),
        ("household_size", "HHSIZE", 8, 1, "collapse", "Current harmonization maps code 8 to size 1."),
        ("household_size", "HHSIZE", ">5", 5, "collapse", "All household sizes above 5 become 5plus."),
        ("household_income", "TotInc", 88888888, None, "drop", "Dropped as invalid/missing income code."),
        ("household_income", "TotInc", 99999999, 695, "collapse", "Current harmonization maps this code to the lowest income band."),
        ("household_income", "TotInc", "<20000", 695, "collapse", "Collapsed to lt_20k."),
        ("household_income", "TotInc", "20000-59999", 697, "collapse", "Collapsed to 20k_to_59k."),
        ("household_income", "TotInc", "60000-99999", 701, "collapse", "Collapsed to 60k_to_99k."),
        ("household_income", "TotInc", ">=100000", 705, "collapse", "Collapsed to 100k_plus."),
        ("prihm", "PRIHM", 9, 0, "collapse", "Current harmonization maps code 9 to 0."),
    ]
    for parameter, source_name, source_value, internal_value, action_type, notes in seed_transform_rows:
        rows.append(
            {
                "section": "seed_transform",
                "parameter": parameter,
                "source_name": source_name,
                "source_value": source_value,
                "internal_name": parameter,
                "internal_value": internal_value,
                "public_name": parameter,
                "public_value": internal_value,
                "action_type": action_type,
                "notes": notes,
            }
        )

    cfstat_mapping = {
        1: 0,
        2: 1,
        4: 1,
        3: 2,
        5: 2,
        7: 3,
        8: 3,
        6: 4,
    }
    for source_value, internal_value in sorted(cfstat_mapping.items()):
        rows.append(
            {
                "section": "seed_transform",
                "parameter": "family_status",
                "source_name": "CFSTAT",
                "source_value": source_value,
                "internal_name": "family_status",
                "internal_value": internal_value,
                "public_name": "family_status",
                "public_value": PUBLIC_VALUE_LABELS["family_status"][internal_value],
                "action_type": "collapse",
                "notes": "CFSTAT collapsed to harmonized family-status role.",
            }
        )

    return pd.DataFrame(rows)
