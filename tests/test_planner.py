import numpy as np

from spatteraug.config import load_config
from spatteraug.geometry import empty_instances
from spatteraug.planner import build_plan


def _config():
    return {
        "seed": 123,
        "classes": {
            "c1": {
                "assets_dir": "/tmp/assets/c1",
                "target_class_id": 1,
                "apply": {"p": 1.0, "conditions": {}},
                "mode": {"type": "whole_file"},
                "intensity": {"n_files": {"dist": "uniform_int", "low": 1, "high": 1}},
                "transform": {"scale": {"dist": "uniform_float", "min": 1.0, "max": 1.0}},
            }
        },
    }


def test_planner_determinism():
    cfg = load_config(_config())
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    instances = empty_instances()
    plan1 = build_plan(rng1, cfg, instances, {"c1": 2})
    plan2 = build_plan(rng2, cfg, instances, {"c1": 2})
    assert plan1 == plan2
