import pandas as pd


def test_build_synth_attr_frame_completes_missing_categories_with_zero_share():
    from synthetic_population_qc.explore.census_compare import build_synth_attr_frame

    households = pd.DataFrame(
        {
            "area": ["da_1", "da_1", "da_2", "da_2"],
            "dwelling_type": [
                "apartment",
                "other_dwelling",
                "apartment",
                "apartment",
            ],
        }
    )

    frame = build_synth_attr_frame(
        "dwelling_type",
        people_df=pd.DataFrame(),
        households_df=households,
    )

    assert len(frame) == 2 * 3

    da_2 = frame.loc[frame["da_code"] == "da_2"].set_index("category")
    assert da_2.loc["single_detached_house", "count"] == 0
    assert da_2.loc["other_dwelling", "count"] == 0
    assert da_2.loc["apartment", "share"] == 1.0
    assert da_2.loc["single_detached_house", "share"] == 0.0
    assert da_2.loc["other_dwelling", "share"] == 0.0
