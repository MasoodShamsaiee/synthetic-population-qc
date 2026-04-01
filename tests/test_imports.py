import pandas as pd


def test_package_imports():
    import synthetic_population_qc
    from synthetic_population_qc import compute_hhtypes, norm_code

    assert hasattr(synthetic_population_qc, "run_synthetic_population_pipeline")
    assert callable(compute_hhtypes)
    assert norm_code(" 24000001.0 ") == "24000001"


def test_compute_hhtypes_smoke():
    from synthetic_population_qc import compute_hhtypes

    df = pd.DataFrame(
        {
            "HID": [0, 0, 1, 1, 1, 2],
            "age": [40, 38, 35, 10, 8, 72],
        }
    )
    out = compute_hhtypes(df)
    assert "hhtype" in out.columns
    assert set(out["hhtype"].tolist()) >= {1, 3}
