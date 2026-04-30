import pandas as pd


def test_largest_remainder_style_enrichment_smoke(tmp_path):
    from synthetic_population_qc.enrichment import enrich_synthetic_population

    syn = pd.DataFrame(
        {
            "sex": [0, 1, 0, 1],
            "prihm": [1, 0, 1, 0],
            "agegrp": [4, 5, 6, 7],
            "area": ["24660001"] * 4,
            "hdgree": [1, 2, 1, 2],
            "lfact": [0, 0, 2, 1],
            "hhsize": [1, 1, 2, 2],
            "totinc": [0, 1, 2, 1],
            "cfstat": [4, 1, 0, 2],
            "HID": [0, 0, 1, 1],
            "age": [30, 28, 45, 16],
            "hhtype": [0, 0, 2, 2],
        }
    )

    raw_root = tmp_path / "data" / "raw" / "census" / "DA scale"
    for sub in ["dwelling char", "housing", "commute", "imm, citiz", "income", "education", "labour", "hh type, size"]:
        (raw_root / sub).mkdir(parents=True, exist_ok=True)

    (raw_root / "dwelling char" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL2 - Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Single-detached house\nCOL3 - Dwelling characteristics / Total - Occupied private dwellings by structural type of dwelling - 100% data / Apartment in a building that has fewer than five storeys\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL2": [1], "COL3": [1]}).to_csv(raw_root / "dwelling char" / "V818MNIvbF_data.csv", index=False)

    (raw_root / "housing" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL1 - Housing - Total Sex / Total - Private households by tenure - 25% sample data / Owner\nCOL2 - Housing - Total Sex / Total - Private households by tenure - 25% sample data / Renter\nCOL4 - Housing - Total Sex / Total - Occupied private dwellings by condominium status - 25% sample data / Condominium\nCOL5 - Housing - Total Sex / Total - Occupied private dwellings by condominium status - 25% sample data / Not condominium\nCOL6 - Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / No bedrooms\nCOL7 - Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 1 bedroom\nCOL8 - Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 2 bedrooms\nCOL9 - Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 3 bedrooms\nCOL10 - Housing - Total Sex / Total - Occupied private dwellings by number of bedrooms - 25% sample data / 4 or more bedrooms\nCOL11 - Housing - Total Sex / Total - Private households by housing suitability - 25% sample data / Suitable\nCOL12 - Housing - Total Sex / Total - Private households by housing suitability - 25% sample data / Not suitable\nCOL13 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1960 or before\nCOL14 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1961 to 1980\nCOL15 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1981 to 1990\nCOL16 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 1991 to 2000\nCOL17 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2001 to 2005\nCOL18 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2006 to 2010\nCOL19 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2011 to 2015\nCOL20 - Housing - Total Sex / Total - Occupied private dwellings by period of construction - 25% sample data / 2016 to 2021\nCOL21 - Housing - Total Sex / Total - Occupied private dwellings by dwelling condition - 25% sample data / Only regular maintenance and minor repairs needed\nCOL22 - Housing - Total Sex / Total - Occupied private dwellings by dwelling condition - 25% sample data / Major repairs needed\nCOL23 - Housing - Total Sex / Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings - 25% sample data / In core need\nCOL24 - Housing - Total Sex / Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings - 25% sample data / Not in core need\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {"COL0": [24660001], "COL1": [1], "COL2": [1], "COL4": [0], "COL5": [2], "COL6": [0], "COL7": [1], "COL8": [1], "COL9": [0], "COL10": [0], "COL11": [2], "COL12": [0], "COL13": [1], "COL14": [1], "COL15": [0], "COL16": [0], "COL17": [0], "COL18": [0], "COL19": [0], "COL20": [0], "COL21": [2], "COL22": [0], "COL23": [0], "COL24": [2]}
    ).to_csv(raw_root / "housing" / "GxvbVaVRu_data.csv", index=False)

    (raw_root / "commute" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL2 - Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Less than 15 minutes ; Both sexes\nCOL3 - Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 15 to 29 minutes ; Both sexes\nCOL4 - Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 30 to 44 minutes ; Both sexes\nCOL5 - Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 45 to 59 minutes ; Both sexes\nCOL6 - Journey to Work - Total Sex / Total - Commuting duration for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / 60 minutes and over ; Both sexes\nCOL7 - Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Car, truck or van ; Both sexes\nCOL8 - Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Public transit ; Both sexes\nCOL9 - Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Walked ; Both sexes\nCOL10 - Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Bicycle ; Both sexes\nCOL11 - Journey to Work - Total Sex / Total - Main mode of commuting for the employed labour force aged 15 years and over with a usual place of work or no fixed workplace address - 25% sample data ; Both sexes / Other method ; Both sexes\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL2": [1], "COL3": [1], "COL4": [0], "COL5": [0], "COL6": [0], "COL7": [1], "COL8": [1], "COL9": [0], "COL10": [0], "COL11": [0]}).to_csv(raw_root / "commute" / "KrBdfPsx1_data.csv", index=False)

    (raw_root / "imm, citiz" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL4 - Immigration - Total Sex / Total - Citizenship for the population in private households - 25% sample data ; Both sexes / Canadian citizens ; Both sexes\nCOL5 - Immigration - Total Sex / Total - Citizenship for the population in private households - 25% sample data ; Both sexes / Not Canadian citizens ; Both sexes\nCOL8 - Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Non-immigrants ; Both sexes\nCOL9 - Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Immigrants ; Both sexes\nCOL10 - Immigration - Total Sex / Total - Immigrant status and period of immigration for the population in private households - 25% sample data ; Both sexes / Non-permanent residents ; Both sexes\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL4": [3], "COL5": [1], "COL8": [3], "COL9": [1], "COL10": [0]}).to_csv(raw_root / "imm, citiz" / "iilmxSwOn_data.csv", index=False)

    (raw_root / "income" / "New Text Document.txt").write_text(
        "\n".join(["COL0 - GEO UID"] + [f"COL{i} - Income - bucket {i}" for i in range(2, 13)]),
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], **{f"COL{i}": [1 if i < 6 else 0] for i in range(2, 13)}}).to_csv(raw_root / "income" / "tEVGMRS0SSexK_data.csv", index=False)

    (raw_root / "education" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL2 - Education - no certificate\nCOL3 - Education - high school\nCOL4 - Education - postsecondary\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL2": [1], "COL3": [0], "COL4": [1]}).to_csv(raw_root / "education" / "udDo1DFAvI_data.csv", index=False)

    (raw_root / "labour" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL2 - Labour - employed\nCOL3 - Labour - unemployed\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL2": [2], "COL3": [1]}).to_csv(raw_root / "labour" / "EVtduVuRucOlsLjw_data.csv", index=False)

    (raw_root / "hh type, size" / "New Text Document.txt").write_text(
        "COL0 - GEO UID\nCOL2 - one family\nCOL3 - non family\nCOL4 - one person\nCOL5 - couple with children\nCOL6 - couple without children\nCOL7 - one parent\nCOL8 - multigenerational\nCOL9 - family size 2\nCOL10 - family size 3\nCOL11 - family size 4\nCOL12 - family size 5+\n",
        encoding="utf-8",
    )
    pd.DataFrame({"COL0": [24660001], "COL2": [1], "COL3": [0], "COL4": [1], "COL5": [0], "COL6": [1], "COL7": [0], "COL8": [0], "COL9": [1], "COL10": [0], "COL11": [0], "COL12": [0]}).to_csv(raw_root / "hh type, size" / "sl5brrkpJ5_data.csv", index=False)

    result = enrich_synthetic_population(syn_with_hh=syn, data_root=tmp_path)

    assert "dwelling_type" in result.enriched_people.columns
    assert "tenure" in result.enriched_people.columns
    assert "citizenship_status" in result.enriched_people.columns
    assert "commute_mode" in result.enriched_people.columns
    assert "hh_key" in result.enriched_households.columns
    assert len(result.assignment_summary) > 0
