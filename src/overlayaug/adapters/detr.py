from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry import Instances


def to_instances(target: Dict[str, Any]) -> Instances:
    boxes = target["boxes"]
    labels = target["labels"]
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    polygons = target.get("polygons")
    if polygons is None:
        polygons = [None] * len(labels)
    return Instances(
        boxes_xyxy=np.asarray(boxes, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        polygons=list(polygons),
        source=np.zeros((len(labels),), dtype=np.int8),
    )


def from_instances(instances: Instances) -> Dict[str, Any]:
    return {
        "boxes": instances.boxes_xyxy,
        "labels": instances.labels,
        "polygons": instances.polygons,
    }
