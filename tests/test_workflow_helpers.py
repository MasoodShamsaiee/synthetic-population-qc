def test_target_count_handles_all_and_fraction():
    from synthetic_population_qc.workflow import target_count

    assert target_count(100, None) == 100
    assert target_count(100, 0.1) == 10


def test_sample_label_formats_fraction():
    from synthetic_population_qc.workflow import sample_label

    assert sample_label(None) == "all"
    assert sample_label(0.25) == "sample0p25"
