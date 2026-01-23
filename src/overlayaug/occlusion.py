from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import cv2

from .geometry import Instances


def ioa_bbox(overlay_box: np.ndarray, box: np.ndarray) -> float:
    x1 = max(overlay_box[0], box[0])
    y1 = max(overlay_box[1], box[1])
    x2 = min(overlay_box[2], box[2])
    y2 = min(overlay_box[3], box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 0:
        return 0.0
    return inter / area


def bbox_covered_fraction(coverage: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, coverage.shape[1]))
    x2 = max(0, min(x2, coverage.shape[1]))
    y1 = max(0, min(y1, coverage.shape[0]))
    y2 = max(0, min(y2, coverage.shape[0]))
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return 0.0
    covered = int(coverage[y1:y2, x1:x2].sum())
    return covered / float(area)


def polygon_covered_fraction(coverage: np.ndarray, polygon: np.ndarray) -> float:
    if polygon.size == 0:
        return 0.0
    x1 = int(np.floor(np.min(polygon[:, 0])))
    y1 = int(np.floor(np.min(polygon[:, 1])))
    x2 = int(np.ceil(np.max(polygon[:, 0])))
    y2 = int(np.ceil(np.max(polygon[:, 1])))
    x1 = max(0, min(x1, coverage.shape[1]))
    x2 = max(0, min(x2, coverage.shape[1]))
    y1 = max(0, min(y1, coverage.shape[0]))
    y2 = max(0, min(y2, coverage.shape[0]))
    if x1 >= x2 or y1 >= y2:
        return 0.0
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted = polygon.copy().astype(np.int32)
    shifted[:, 0] -= x1
    shifted[:, 1] -= y1
    cv2.fillPoly(mask, [shifted], 1)
    covered = (coverage[y1:y2, x1:x2] & (mask > 0)).sum()
    area = mask.sum()
    if area == 0:
        return 0.0
    return float(covered) / float(area)


def drop_occluded_instances(
    instances: Instances,
    coverage: Optional[np.ndarray],
    metric: str,
    drop_threshold: float,
    min_box_area_px: int,
    base_only: bool,
) -> Instances:
    if instances.boxes_xyxy.shape[0] == 0:
        return instances
    keep = np.ones(instances.boxes_xyxy.shape[0], dtype=bool)
    for idx, box in enumerate(instances.boxes_xyxy):
        if base_only and instances.source[idx] != 0:
            continue
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area < min_box_area_px:
            continue
        if metric == "ioa_bbox":
            raise ValueError("ioa_bbox requires per-overlay boxes")
        if coverage is None:
            continue
        if metric == "alpha_in_bbox":
            occluded = bbox_covered_fraction(coverage, box)
        elif metric == "alpha_in_polygon":
            poly = instances.polygons[idx]
            if poly is None:
                occluded = bbox_covered_fraction(coverage, box)
            else:
                occluded = polygon_covered_fraction(coverage, poly)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        if occluded >= drop_threshold:
            keep[idx] = False
    return Instances(
        boxes_xyxy=instances.boxes_xyxy[keep],
        labels=instances.labels[keep],
        polygons=[p for p, k in zip(instances.polygons, keep) if k],
        source=instances.source[keep],
    )
