from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .assets import AssetStore
from .config import SpatterAugConfig, load_config
from .executor import execute_plan
from .geometry import Instances
from .planner import build_plan


class SpatterAugmentor:
    def __init__(self, config: dict | str, asset_store: AssetStore) -> None:
        self.config: SpatterAugConfig = load_config(config)
        self.asset_store = asset_store
        self._class_file_counts = {
            name: self.asset_store.file_count(name) for name in self.asset_store.classes()
        }
        self._epoch = 0
        self._rank = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_rank(self, rank: int) -> None:
        self._rank = int(rank)

    def _make_rng(self, index: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.config.seed, self._epoch, index, self._rank])
        return np.random.default_rng(ss)

    def __call__(
        self, image: np.ndarray, target: dict | Instances, index: int
    ) -> Tuple[np.ndarray, dict | Instances, Optional[dict]]:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be numpy array")
        rng = self._make_rng(index)
        if isinstance(target, Instances):
            instances = target
        elif isinstance(target, dict):
            boxes = target["boxes"]
            labels = target["labels"]
            polygons = target.get("polygons")
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().numpy()
            if hasattr(labels, "detach"):
                labels = labels.detach().cpu().numpy()
            if polygons is None:
                polygons = [None] * len(labels)
            instances = Instances(
                boxes_xyxy=np.asarray(boxes, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int64),
                polygons=list(polygons),
                source=np.zeros((len(labels),), dtype=np.int8),
            )
        else:
            raise ValueError("target must be dict or Instances")

        plan = build_plan(
            rng,
            self.config,
            instances,
            self._class_file_counts,
        )
        out_image, out_instances, debug = execute_plan(
            rng,
            self.config,
            self.asset_store,
            image,
            instances,
            plan,
        )
        if isinstance(target, Instances):
            return out_image, out_instances, debug
        return (
            out_image,
            {
                "boxes": out_instances.boxes_xyxy,
                "labels": out_instances.labels,
                "polygons": out_instances.polygons,
            },
            debug,
        )
