# Data Contracts

## Purpose

This package still carries notebook-derived workflow logic, so explicit contracts are especially important. The items below capture the minimum expected shapes for the main pipeline tables.

## Synthetic individuals table

Typical output columns:

- `area`
- `sex`
- `age` or `agegrp`
- optional household columns such as `HID`
- optional socio-demographic collapsed variables such as `lfact`, `hdgree`, `totinc`, `cfstat`

Operational expectations:

- one row per person
- `area` identifies the DA
- household assignment functions expect `age` or `agegrp`

## Census profile table

Required fields used repeatedly in workflows:

- `geocode`
- `variableId`
- `total`

Optional but used in some summaries:

- `totalMale`
- `totalFemale`

Operational expectations:

- one row per geocode-variable pair
- suppressed values may appear as `x`, `F`, or `..`

## Household assignment contract

`assign_households_using_census` expects:

- a synthetic-individual table with an area column
- a census profile table
- household-size and total-household variable-id mappings

It returns:

- synthetic population with `HID` and `hhtype`
- a DA-level summary table

## Workflow stability rule

Any future cleanup of `notebook_functions.py` should preserve these external table contracts or introduce an explicit migration note.
