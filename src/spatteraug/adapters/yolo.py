from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..geometry import (
    Instances,
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
    denormalize_boxes_xywh,
    denormalize_polygons,
    normalize_boxes_xyxy,
    normalize_polygons,
)


def to_instances(target: Dict[str, Any], width: int, height: int) -> Instances:
    boxes = np.asarray(target.get("bboxes", []), dtype=np.float32)
    labels = np.asarray(target.get("labels", []), dtype=np.int64)
    polygons = target.get("polygons") or [None] * len(labels)
    boxes_px = bbox_xywh_to_xyxy(denormalize_boxes_xywh(boxes, width, height))
    polygons_px = [
        None if poly is None else poly * np.array([width, height], dtype=np.float32)
        for poly in polygons
    ]
    return Instances(
        boxes_xyxy=boxes_px,
        labels=labels,
        polygons=list(polygons_px),
        source=np.zeros((len(labels),), dtype=np.int8),
    )


def from_instances(instances: Instances, width: int, height: int) -> Dict[str, Any]:
    boxes_norm = normalize_boxes_xyxy(instances.boxes_xyxy, width, height)
    boxes_xywh = bbox_xyxy_to_xywh(boxes_norm)
    polygons_norm = normalize_polygons(instances.polygons, width, height)
    polygons = [
        None if poly is None else poly for poly in polygons_norm
    ]
    return {
        "bboxes": boxes_xywh,
        "labels": instances.labels,
        "polygons": polygons,
    }
