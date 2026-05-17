import importlib.util
import sys
from pathlib import Path


def load_run_experiment_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_experiment"] = module
    spec.loader.exec_module(module)
    return module


def test_choose_profile_by_gpu_memory():
    module = load_run_experiment_module()

    assert module.choose_profile([]) == "cpu"
    assert module.choose_profile([("NVIDIA T4", 15360)]) == "medium"
    assert module.choose_profile([("NVIDIA RTX 4090", 24564)]) == "large"
    assert module.choose_profile([("NVIDIA A800-SXM4-80GB", 81920)]) == "xlarge"


def test_profile_shapes_are_divisible_by_heads():
    module = load_run_experiment_module()

    for profile in module.PROFILES.values():
        assert profile.hidden_size % profile.num_heads == 0
        assert profile.pretrain_lr == 1e-5
        assert profile.finetune_lr == 1e-5
