"""Household and person enrichment against DA-side census context tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_population_qc.context_tables import load_context_tables


HOUSEHOLD_ATTR_LABELS = {
    "dwelling_type": {
        "single_detached": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Single-detached house",
        "semi_detached": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Semi-detached house",
        "row_house": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Row house",
        "duplex_apartment": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Apartment or flat in a duplex",
        "lowrise_apartment": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Apartment in a building that has fewer than five storeys",
        "highrise_apartment": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Apartment in a building that has five or more storeys",
        "other_single_attached": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Other single-attached house",
        "movable_dwelling": "Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Movable dwelling",
    },
    "tenure": {
        "owner": "Housing - Total Sex / Total - Private households by tenure - 25% sample data / Owner",
        "renter": "Housing - Total Sex / Total - Private households by tenure - 25% sample data / Renter",
        "band_housing": "Housing - Total Sex / Total - Private households by tenure - 25% sample data / Dwelling provided by the local government, First Nation or Indian band",
    },
    "condo_status": {
        "condominium": "Housing - Total Sex / Total - Occupied private dwellings by condominium status - 25% sample data / Condominium",
        "not_condominium": "Housing - Total Sex / Total - Occupied private dwellings by condominium status - 25% sample data / Not condominium",
    },
    "bedrooms": {
        "no_bedrooms": "Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / No bedrooms",
        "1_bedroom": "Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 1 bedroom",
        "2_bedrooms": "Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 2 bedrooms",
        "3_bedrooms": "Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 3 bedrooms",
        "4plus_bedrooms": "Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 4 or more bedrooms",
    },
    "housing_suitability": {
        "suitable": "Housing - Total Sex / Total - Private households by housing suitability - 25% sample data / Suitable",
        "not_suitable": "Housing - Total Sex / Total - Private households by housing suitability - 25% sample data / Not suitable",
    },
    "period_built": {
        "1960_or_before": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1960 or before",
        "1961_to_1980": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1961 to 1980",
        "1981_to_1990": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1981 to 1990",
        "1991_to_2000": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1991 to 2000",
        "2001_to_2005": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2001 to 2005",
        "2006_to_2010": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2006 to 2010",
        "2011_to_2015": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2011 to 2015",
        "2016_to_2021": "Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2016 to 2021",
    },
    "dwelling_condition": {
        "regular_or_minor_repairs": "Housing - Total Sex / Total - Occupied private dwellings by dwelling condition - 25% sample data / Only regular maintenance and minor repairs needed",
        "major_repairs_needed": "Housing - Total Sex / Total - Occupied private dwellings by dwelling condition - 25% sample data / Major repairs needed",
    },
    "core_housing_need": {
        "in_core_need": "Housing - Total Sex / Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings - 25% sample data / In core need",
        "not_in_core_need": "Housing - Total Sex / Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings - 25% sample data / Not in core need",
    },
}

PERSON_ATTR_LABELS = {
    "citizenship_status": {
        "canadian_citizen": "Immigration - Total Sex / Total - Citizenship for the population in private households - 25% sample data ; Both sexes / Canadian citizens ; Both sexes",
        "not_canadian_citizen": "Immigration - Total Sex / Total - Citizenship for the population in private households - 25% sample data ; Both sexes / Not Canadian citizens ; Both sexes",
    },
    "immigrant_status": {
        "non_immigrant": "Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Non-immigrants ; Both sexes",
        "immigrant": "Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Immigrants ; Both sexes",
        "non_permanent_resident": "Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Non-permanent residents ; Both sexes",
    },
    "after_tax_income_band": {
        "under_10k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / Under $10,000 (including loss) ; Both sexes",
        "10k_to_19k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $10,000 to $19,999 ; Both sexes",
        "20k_to_29k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $20,000 to $29,999 ; Both sexes",
        "30k_to_39k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $30,000 to $39,999 ; Both sexes",
        "40k_to_49k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $40,000 to $49,999 ; Both sexes",
        "50k_to_59k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $50,000 to $59,999 ; Both sexes",
        "60k_to_69k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $60,000 to $69,999 ; Both sexes",
        "70k_to_79k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $70,000 to $79,999 ; Both sexes",
        "80k_to_89k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $80,000 to $89,999 ; Both sexes",
        "90k_to_99k": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $90,000 to $99,999 ; Both sexes",
        "100k_plus": "Income - Total Sex / Total - After-tax income groups in 2020 for the population aged 15 years and over in private households - 100% data ; Both sexes / With after-tax income ; Both sexes / $100,000 and over ; Both sexes",
    },
    "education_detail": {
        "no_certificate": "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data ; Both sexes / No certificate, diploma or degree ; Both sexes",
        "high_school": "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data ; Both sexes / High (secondary) school diploma or equivalency certificate ; Both sexes",
        "postsecondary": "Education - Total Sex / Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data ; Both sexes / Postsecondary certificate, diploma or degree ; Both sexes",
    },
    "commute_mode": {
        "car_truck_van": "Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Car, truck or van ; Both sexes",
        "public_transit": "Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Public transit ; Both sexes",
        "walked": "Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Walked ; Both sexes",
        "bicycle": "Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Bicycle ; Both sexes",
        "other_method": "Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Other method ; Both sexes",
    },
    "commute_duration": {
        "lt_15_min": "Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Less than 15 minutes ; Both sexes",
        "15_to_29_min": "Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 15 to 29 minutes ; Both sexes",
        "30_to_44_min": "Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 30 to 44 minutes ; Both sexes",
        "45_to_59_min": "Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 45 to 59 minutes ; Both sexes",
        "60plus_min": "Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 60 minutes and over ; Both sexes",
    },
}


@dataclass
class EnrichmentResult:
    """Person/household tables plus a compact summary of the enrichment pass."""
    enriched_people: pd.DataFrame
    enriched_households: pd.DataFrame
    assignment_summary: pd.DataFrame


def _largest_remainder(counts: dict[str, float], target_total: int) -> dict[str, int]:
    """Integerize weighted category targets while preserving the requested total."""
    keys = list(counts.keys())
    vals = {k: max(0.0, float(counts.get(k, 0.0))) for k in keys}
    total = sum(vals.values())
    if target_total <= 0:
        return {k: 0 for k in keys}
    if total <= 0:
        base = {k: 0 for k in keys}
        if keys:
            base[keys[0]] = target_total
        return base
    scaled = {k: target_total * vals[k] / total for k in keys}
    floors = {k: int(np.floor(v)) for k, v in scaled.items()}
    rem = target_total - sum(floors.values())
    fracs = sorted(((k, scaled[k] - floors[k]) for k in keys), key=lambda x: x[1], reverse=True)
    for i in range(rem):
        floors[fracs[i % len(fracs)][0]] += 1
    return floors


def _extract_counts(row: pd.Series | None, labels: dict[str, str]) -> dict[str, float]:
    """Pull labeled context-table counts out of one DA row."""
    if row is None or row.empty:
        return {k: 0.0 for k in labels}
    out = {}
    for key, col in labels.items():
        out[key] = float(pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0] or 0.0)
    return out


def _choose_fallback_row(table: pd.DataFrame, da_code: str) -> pd.Series | None:
    """Choose a broader fallback row when a DA-level row is unavailable."""
    if table.empty or "da_code" not in table.columns:
        return None
    exact = table.loc[table["da_code"].astype(str) == str(da_code)]
    if not exact.empty:
        return exact.iloc[0]
    prefix = str(da_code)[:4]
    prefix_rows = table.loc[table["da_code"].astype(str).str.startswith(prefix, na=False)]
    if not prefix_rows.empty:
        numeric = prefix_rows.select_dtypes(include=[np.number])
        return numeric.mean(numeric_only=True)
    return table.select_dtypes(include=[np.number]).mean(numeric_only=True)


def _weighted_assign_exact(
    df: pd.DataFrame,
    *,
    counts: dict[str, int],
    score_map: dict[str, pd.Series],
    rng: np.random.Generator,
    out_col: str,
) -> pd.Series:
    """Assign exact category counts by weighted sampling without changing totals."""
    remaining = list(df.index)
    assigned = pd.Series(index=df.index, dtype="object")
    for category, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        n = int(min(max(n, 0), len(remaining)))
        if n == 0:
            continue
        scores = score_map[category].loc[remaining].astype(float).to_numpy()
        scores = np.where(np.isfinite(scores), scores, 0.0)
        scores = np.clip(scores, 1e-9, None)
        probs = scores / scores.sum()
        chosen = rng.choice(np.array(remaining, dtype=object), size=n, replace=False, p=probs)
        assigned.loc[list(chosen)] = category
        chosen_set = set(chosen.tolist())
        remaining = [idx for idx in remaining if idx not in chosen_set]
    if remaining:
        fallback = max(counts, key=counts.get) if counts else None
        assigned.loc[remaining] = fallback
    return assigned.rename(out_col)


def _build_household_table(syn_with_hh: pd.DataFrame) -> pd.DataFrame:
    """Collapse person-linked records into one row per synthetic household."""
    hh_people = syn_with_hh.loc[syn_with_hh["HID"] != -1].copy()
    hh_people["area"] = hh_people["area"].astype(str)
    hh_people["hh_key"] = hh_people["area"] + "__" + hh_people["HID"].astype(str)
    hh = (
        hh_people.groupby("hh_key", sort=False)
        .agg(
            area=("area", "first"),
            HID=("HID", "first"),
            hh_size=("hh_key", "size"),
            hhtype=("hhtype", "first"),
            n_children=("age", lambda s: int((pd.to_numeric(s, errors="coerce") < 18).sum())),
            n_adults=("age", lambda s: int((pd.to_numeric(s, errors="coerce") >= 18).sum())),
            mean_age=("age", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
            max_totinc=("totinc", lambda s: int(pd.to_numeric(s, errors="coerce").max())),
        )
        .reset_index()
    )
    hh["is_family"] = hh["hhtype"].isin([0, 1, 2]).astype(int)
    hh["is_singleton"] = (hh["hh_size"] == 1).astype(int)
    hh["is_large"] = (hh["hh_size"] >= 4).astype(int)
    return hh


def _household_score_maps(hh: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Build heuristic household-side score maps for extension attributes."""
    size = pd.to_numeric(hh["hh_size"], errors="coerce").fillna(1)
    single = hh["is_singleton"]
    large = hh["is_large"]
    family = hh["is_family"]

    return {
        "dwelling_type": {
            "single_detached": 1 + 1.8 * large + 1.0 * family + 0.8 * (hh["hhtype"] == 0),
            "semi_detached": 1 + 1.2 * family + 0.6 * (size >= 3),
            "row_house": 1 + 0.8 * family + 0.8 * (size >= 3),
            "duplex_apartment": 1 + 0.6 * family + 0.4 * (size <= 3),
            "lowrise_apartment": 1 + 1.2 * single + 0.7 * (size <= 2),
            "highrise_apartment": 1 + 1.5 * single + 0.8 * (size <= 2),
            "other_single_attached": 1 + 0.8 * family,
            "movable_dwelling": pd.Series(1.0, index=hh.index),
        },
        "tenure": {
            "owner": 1 + 1.2 * family + 1.3 * large + 0.8 * (hh["hhtype"] == 0),
            "renter": 1 + 1.5 * single + 0.8 * (size <= 2),
            "band_housing": pd.Series(1.0, index=hh.index),
        },
        "condo_status": {
            "condominium": 1 + 1.2 * single + 0.8 * (size <= 2),
            "not_condominium": 1 + 0.6 * family + 0.6 * large,
        },
        "bedrooms": {
            "no_bedrooms": 1 + 1.8 * single * (size == 1),
            "1_bedroom": 1 + 1.4 * single + 0.4 * (size == 2),
            "2_bedrooms": 1 + 1.0 * (size == 2) + 0.6 * (size == 3),
            "3_bedrooms": 1 + 1.0 * (size == 3) + 0.8 * (size == 4),
            "4plus_bedrooms": 1 + 1.5 * large,
        },
        "housing_suitability": {
            "suitable": 1 + 0.2 * single,
            "not_suitable": 1 + 1.8 * large + 0.8 * (size == 3),
        },
        "period_built": {k: pd.Series(1.0, index=hh.index) for k in HOUSEHOLD_ATTR_LABELS["period_built"]},
        "dwelling_condition": {
            "regular_or_minor_repairs": pd.Series(1.0, index=hh.index),
            "major_repairs_needed": 1 + 0.7 * large,
        },
        "core_housing_need": {
            "in_core_need": 1 + 0.8 * single + 0.8 * large + 0.4 * (hh["max_totinc"] <= 1),
            "not_in_core_need": 1 + 0.5 * family + 0.5 * (hh["max_totinc"] >= 2),
        },
    }


def _person_score_maps(persons: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Build heuristic person-side score maps for extension attributes."""
    age = pd.to_numeric(persons["age"], errors="coerce").fillna(0)
    sex = pd.to_numeric(persons["sex"], errors="coerce").fillna(0)
    lfact = pd.to_numeric(persons["lfact"], errors="coerce").fillna(2)
    income = pd.to_numeric(persons["totinc"], errors="coerce").fillna(0)
    employed = (lfact == 0).astype(int)

    income_scores = {}
    for idx, key in enumerate(PERSON_ATTR_LABELS["after_tax_income_band"]):
        target = idx / 2.5
        income_scores[key] = 1 + np.exp(-np.abs(income - target))

    return {
        "citizenship_status": {
            "canadian_citizen": 1 + 0.2 * (age >= 18),
            "not_canadian_citizen": 1 + 0.5 * (age >= 18),
        },
        "immigrant_status": {
            "non_immigrant": 1 + 0.2 * (age >= 18),
            "immigrant": 1 + 0.5 * (age >= 18),
            "non_permanent_resident": 1 + 0.3 * (age >= 18),
        },
        "after_tax_income_band": income_scores,
        "education_detail": {
            "no_certificate": 1 + 0.8 * ((age >= 25) & (age <= 34)) + 0.3 * (income == 0),
            "high_school": 1 + 0.5 * ((age >= 25) & (age <= 54)),
            "postsecondary": 1 + 0.5 * ((age >= 25) & (age <= 64)) + 0.6 * (income >= 2),
        },
        "commute_mode": {
            "car_truck_van": 1 + 1.0 * employed + 0.4 * (income >= 2),
            "public_transit": 1 + 1.0 * employed + 0.2 * (income <= 1),
            "walked": 1 + 0.6 * employed + 0.2 * (age < 35),
            "bicycle": 1 + 0.4 * employed + 0.2 * (age < 45),
            "other_method": pd.Series(1.0, index=persons.index),
        },
        "commute_duration": {
            "lt_15_min": 1 + 0.4 * employed,
            "15_to_29_min": 1 + 0.5 * employed,
            "30_to_44_min": 1 + 0.5 * employed + 0.1 * sex,
            "45_to_59_min": 1 + 0.3 * employed,
            "60plus_min": 1 + 0.3 * employed,
        },
    }


def enrich_synthetic_population(
    *,
    syn_with_hh: pd.DataFrame,
    data_root: str | Path,
    random_seed: int = 42,
) -> EnrichmentResult:
    """Assign extension attributes to synthetic people and households by DA."""
    rng = np.random.default_rng(int(random_seed))
    context = load_context_tables(Path(data_root))

    people = syn_with_hh.copy()
    people["area"] = people["area"].astype(str)
    people["person_id"] = np.arange(len(people))
    people["hh_key"] = np.where(
        people["HID"] != -1,
        people["area"] + "__" + people["HID"].astype(str),
        people["area"] + "__unassigned__" + people["person_id"].astype(str),
    )

    households = _build_household_table(people)
    hh_score_maps = _household_score_maps(households)
    person_score_maps = _person_score_maps(people)

    summaries: list[dict] = []
    enriched_households = households.copy()
    for attr, labels in HOUSEHOLD_ATTR_LABELS.items():
        table_name = "dwelling_characteristics" if attr == "dwelling_type" else "housing"
        source = context.get(table_name, pd.DataFrame())
        assigned_parts = []
        for da_code, hh_da in households.groupby("area", sort=False):
            row = _choose_fallback_row(source, da_code)
            counts = _largest_remainder(_extract_counts(row, labels), len(hh_da))
            assigned = _weighted_assign_exact(
                hh_da,
                counts=counts,
                score_map=hh_score_maps[attr],
                rng=rng,
                out_col=attr,
            )
            assigned_parts.append(assigned)
            summaries.append(
                {
                    "level": "household",
                    "attribute": attr,
                    "da_code": da_code,
                    "n_items": len(hh_da),
                    "categories": "|".join(f"{k}:{v}" for k, v in counts.items()),
                }
            )
        enriched_households[attr] = pd.concat(assigned_parts).reindex(enriched_households.index)

    enriched_people = people.merge(
        enriched_households[
            [
                "hh_key",
                "dwelling_type",
                "tenure",
                "condo_status",
                "bedrooms",
                "housing_suitability",
                "period_built",
                "dwelling_condition",
                "core_housing_need",
            ]
        ],
        on="hh_key",
        how="left",
    )

    for attr, labels in PERSON_ATTR_LABELS.items():
        if attr in {"commute_mode", "commute_duration"}:
            source = context.get("commute", pd.DataFrame())
            eligible_mask = pd.to_numeric(enriched_people["lfact"], errors="coerce").fillna(2) == 0
        elif attr in {"citizenship_status", "immigrant_status"}:
            source = context.get("immigration_citizenship", pd.DataFrame())
            eligible_mask = pd.Series(True, index=enriched_people.index)
        elif attr == "after_tax_income_band":
            source = context.get("income_detailed", pd.DataFrame())
            eligible_mask = pd.to_numeric(enriched_people["age"], errors="coerce").fillna(0) >= 15
        elif attr == "education_detail":
            source = context.get("education_detailed", pd.DataFrame())
            age = pd.to_numeric(enriched_people["age"], errors="coerce").fillna(0)
            eligible_mask = (age >= 25) & (age <= 64)
        else:
            source = pd.DataFrame()
            eligible_mask = pd.Series(True, index=enriched_people.index)

        result = pd.Series(index=enriched_people.index, dtype="object")
        for da_code, people_da in enriched_people.groupby("area", sort=False):
            eligible_idx = people_da.index[eligible_mask.loc[people_da.index]]
            if len(eligible_idx) == 0:
                continue
            row = _choose_fallback_row(source, da_code)
            counts = _largest_remainder(_extract_counts(row, labels), len(eligible_idx))
            assigned = _weighted_assign_exact(
                enriched_people.loc[eligible_idx],
                counts=counts,
                score_map={k: person_score_maps[attr].get(k, pd.Series(1.0, index=enriched_people.index)) for k in labels},
                rng=rng,
                out_col=attr,
            )
            result.loc[eligible_idx] = assigned
            summaries.append(
                {
                    "level": "person",
                    "attribute": attr,
                    "da_code": da_code,
                    "n_items": len(eligible_idx),
                    "categories": "|".join(f"{k}:{v}" for k, v in counts.items()),
                }
            )
        enriched_people[attr] = result

    enriched_people["workforce_participant"] = pd.to_numeric(enriched_people["lfact"], errors="coerce").fillna(2) == 0
    enriched_people["has_housing_enrichment"] = enriched_people["dwelling_type"].notna()

    assignment_summary = pd.DataFrame(summaries)
    return EnrichmentResult(
        enriched_people=enriched_people.drop(columns=["person_id"], errors="ignore"),
        enriched_households=enriched_households,
        assignment_summary=assignment_summary,
    )
