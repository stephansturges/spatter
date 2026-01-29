from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import ClassConfig, SpatterAugConfig
from .geometry import Instances


@dataclass
class PasteOp:
    class_name: str
    class_id: int
    mode: str
    asset_file_id: int
    asset_instance_ids: List[int]
    transform_params: Dict[str, float]
    placement_params: Dict[str, float]


def _check_conditions(instances: Instances, conditions: Dict[str, Optional[int]]) -> bool:
    only_has = conditions.get("only_if_base_has_class")
    only_lacks = conditions.get("only_if_base_lacks_class")
    labels = set(instances.labels.tolist())
    if only_has is not None and only_has not in labels:
        return False
    if only_lacks is not None and only_lacks in labels:
        return False
    return True


def _select_mode(mode_cfg: ClassConfig, rng: np.random.Generator) -> str:
    mode_type = mode_cfg.mode.type
    if mode_type != "mixed":
        return mode_type
    weights = mode_cfg.mode.mixed_weights or {}
    options = list(weights.keys())
    probs = np.array([weights[k] for k in options], dtype=np.float32)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(options)) / len(options)
    return str(rng.choice(options, p=probs))


def build_plan(
    rng: np.random.Generator,
    config: SpatterAugConfig,
    instances: Instances,
    class_file_counts: Dict[str, int],
) -> List[PasteOp]:
    ops: List[PasteOp] = []
    for class_name in sorted(config.classes.keys()):
        cls_cfg = config.classes[class_name]
        if not _check_conditions(instances, cls_cfg.apply.conditions):
            continue
        if rng.random() > cls_cfg.apply.p:
            continue
        mode = _select_mode(cls_cfg, rng)
        n_files = cls_cfg.intensity.n_files.sample(rng)
        n_files = max(0, n_files)
        file_count = class_file_counts.get(class_name, 0)
        if file_count == 0:
            continue
        file_ids = rng.choice(file_count, size=min(n_files, file_count), replace=False)
        for file_id in file_ids:
            if mode == "whole_file":
                ops.append(
                    PasteOp(
                        class_name=class_name,
                        class_id=cls_cfg.target_class_id,
                        mode=mode,
                        asset_file_id=int(file_id),
                        asset_instance_ids=[],
                        transform_params={},
                        placement_params={},
                    )
                )
            else:
                k_dist = cls_cfg.intensity.k_instances_per_file
                if k_dist is None:
                    continue
                k = k_dist.sample(rng)
                instance_ids = list(range(k))
                ops.append(
                    PasteOp(
                        class_name=class_name,
                        class_id=cls_cfg.target_class_id,
                        mode="instance",
                        asset_file_id=int(file_id),
                        asset_instance_ids=instance_ids,
                        transform_params={},
                        placement_params={},
                    )
                )
    if config.global_cfg.strategy == "budgeted":
        rng.shuffle(ops)
    if len(ops) > config.global_cfg.max_total_pastes:
        ops = ops[: config.global_cfg.max_total_pastes]
    return ops
