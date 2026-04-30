import importlib


def test_package_imports_expose_supported_workflow():
    import synthetic_population_qc
    from synthetic_population_qc import (
        RawDataRoots,
        WorkflowSettings,
        default_geometry_dir,
        load_run_bundle,
        project_root,
        run_energy_population_workflow,
    )

    assert callable(run_energy_population_workflow)
    assert callable(load_run_bundle)
    assert callable(project_root)
    assert callable(default_geometry_dir)
    assert RawDataRoots.__name__ == "RawDataRoots"
    assert WorkflowSettings.__name__ == "WorkflowSettings"
    assert not hasattr(synthetic_population_qc, "compute_hhtypes")


def test_clean_support_modules_are_importable():
    context_tables = importlib.import_module("synthetic_population_qc.context_tables")
    energy_workflow = importlib.import_module("synthetic_population_qc.energy_workflow")
    reporting = importlib.import_module("synthetic_population_qc.reporting")
    seed_transforms = importlib.import_module("synthetic_population_qc.seed_transforms")
    seed_preparation = importlib.import_module("synthetic_population_qc.seed_preparation")
    utils = importlib.import_module("synthetic_population_qc.utils")
    workflow_inputs = importlib.import_module("synthetic_population_qc.workflow_inputs")

    assert hasattr(context_tables, "load_context_tables")
    assert hasattr(energy_workflow, "run_full_energy_aware_workflow")
    assert hasattr(reporting, "build_results_summary")
    assert hasattr(seed_preparation, "prepare_person_seed")
    assert hasattr(seed_transforms, "probabilistic_sampling")
    assert hasattr(utils, "build_reference_census")
    assert hasattr(workflow_inputs, "build_workflow_input_contract")
