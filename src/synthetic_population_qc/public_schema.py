"""Canonical public names and labels for workflow outputs."""

from __future__ import annotations

import pandas as pd


PUBLIC_ATTR_RENAMES: dict[str, str] = {
    "sex_native": "sex",
    "agegrp_core": "age_group",
    "hdgree_core": "education_level",
    "lfact_core": "labour_force_status",
    "totinc_core": "household_income",
    "cfstat_core": "family_status",
    "hhsize_core": "household_size",
    "household_size_native": "household_size",
    "household_type_native": "household_type",
}

PUBLIC_VALUE_LABELS: dict[str, dict[object, object]] = {
    "education_level": {
        1684: "no_certificate_or_diploma",
        1685: "high_school_or_equivalent",
        1686: "postsecondary_certificate_diploma_or_degree",
    },
    "labour_force_status": {
        1867: "employed",
        1868: "unemployed",
        1869: "not_in_labour_force",
    },
    "household_income": {
        695: "lt_20k",
        697: "20k_to_59k",
        701: "60k_to_99k",
        705: "100k_plus",
    },
    "family_status": {
        0: "couple_without_children_family_member",
        1: "couple_with_children_family_member",
        2: "one_parent_family_member",
        3: "non_census_family_person",
        4: "one_person_household_person",
    },
}

AGE_GROUP_BOUNDS: dict[int, tuple[int, int | None]] = {
    10: (0, 4),
    11: (5, 14),
    12: (15, 24),
    14: (25, 34),
    15: (35, 39),
    16: (40, 44),
    17: (45, 49),
    18: (50, 54),
    19: (55, 59),
    20: (60, 64),
    21: (65, 69),
    22: (70, 74),
    23: (75, 79),
    25: (80, 84),
    26: (85, None),
}

AGE_GROUP_SCHEME_BREAKS: dict[str, tuple[int, ...]] = {
    "default_15": (0, 5, 15, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85),
    "default_18": (0, 5, 15, 25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85),
    "coarse_10": (0, 15, 25, 35, 45, 55, 65, 75, 80, 85),
    "broad_6": (0, 15, 25, 45, 65, 85),
}


def public_attr_name(name: str) -> str:
    """Return the supported semantic name for a workflow attribute."""
    return PUBLIC_ATTR_RENAMES.get(str(name), str(name))


def public_attr_list(names: list[str]) -> list[str]:
    """Normalize a list of attribute names to the supported public schema."""
    return [public_attr_name(name) for name in names]


def public_conditioning_cols(names: list[str]) -> list[str]:
    """Normalize conditioning-column names to their public equivalents."""
    return [public_attr_name(name) for name in names]


def _format_age_label(start: int, end: int | None) -> str:
    """Format one public age-group label from bucket boundaries."""
    if end is None:
        return f"{start}_plus"
    return f"{start}_to_{end}"


def age_group_value_labels(
    *,
    scheme: str = "default_15",
    custom_breaks: list[int] | tuple[int, ...] | None = None,
) -> dict[int, str]:
    """Map harmonized age-bucket codes to public age-group labels."""
    if custom_breaks is not None:
        breaks = tuple(int(x) for x in custom_breaks)
    else:
        if scheme not in AGE_GROUP_SCHEME_BREAKS:
            raise ValueError(f"Unsupported age grouping scheme: {scheme}")
        breaks = AGE_GROUP_SCHEME_BREAKS[scheme]
    if len(breaks) < 2:
        raise ValueError("Age grouping requires at least two breakpoints.")
    labels: dict[int, str] = {}
    windows = list(breaks)
    for code, bounds in AGE_GROUP_BOUNDS.items():
        lo, hi = bounds
        match_idx = None
        for idx, start in enumerate(windows):
            next_start = windows[idx + 1] if idx + 1 < len(windows) else None
            bucket_end = (next_start - 1) if next_start is not None else None
            if lo < start:
                continue
            if bucket_end is None:
                if hi is None or hi >= start:
                    match_idx = idx
                    break
            elif hi is not None and hi <= bucket_end:
                match_idx = idx
                break
        if match_idx is None:
            raise ValueError(
                f"Custom age breaks {breaks} cannot represent existing harmonized bucket {code} ({lo}, {hi})."
            )
        start = windows[match_idx]
        next_start = windows[match_idx + 1] if match_idx + 1 < len(windows) else None
        labels[code] = _format_age_label(start, (next_start - 1) if next_start is not None else None)
    return labels


def public_value_label(
    attr_name: str,
    value: object,
    *,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> object:
    """Convert one internal value into its supported public label."""
    if pd.isna(value):
        return value
    public_name = public_attr_name(attr_name)
    mapping = (
        age_group_value_labels(scheme=age_group_scheme, custom_breaks=age_group_breaks)
        if public_name == "age_group"
        else PUBLIC_VALUE_LABELS.get(public_name)
    )
    if not mapping:
        return value
    if value in mapping:
        return mapping[value]
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        normalized = int(numeric) if float(numeric).is_integer() else float(numeric)
        if normalized in mapping:
            return mapping[normalized]
    return value


def public_value_series(
    attr_name: str,
    series: pd.Series,
    *,
    age_group_scheme: str = "default_15",
    age_group_breaks: list[int] | tuple[int, ...] | None = None,
) -> pd.Series:
    """Convert a Series from internal codes into public semantic labels."""
    public_name = public_attr_name(attr_name)
    mapping = (
        age_group_value_labels(scheme=age_group_scheme, custom_breaks=age_group_breaks)
        if public_name == "age_group"
        else PUBLIC_VALUE_LABELS.get(public_name)
    )
    if not mapping:
        return series
    mapped = series.map(
        lambda value: public_value_label(
            public_name,
            value,
            age_group_scheme=age_group_scheme,
            age_group_breaks=age_group_breaks,
        )
    )
    return mapped.astype("object")
