from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .assets import AssetLabel, AssetStore
from .composite import blend_roi, clip_patch, compute_iou, update_coverage
from .config import ClassConfig, SpatterAugConfig
from .geometry import (
    Instances,
    boxes_from_polygons,
    clip_boxes_xyxy,
    filter_small_boxes,
    flip_boxes,
    flip_polygons,
    scale_boxes,
    scale_polygons,
    translate_boxes,
    translate_polygons,
)
from .occlusion import drop_occluded_instances
from .planner import PasteOp


@dataclass
class OverlayPatch:
    rgba: np.ndarray
    bbox: np.ndarray
    polygon: Optional[np.ndarray]


def _extract_patch(
    image_rgba: np.ndarray, label: AssetLabel, cfg: ClassConfig
) -> OverlayPatch:
    h, w = image_rgba.shape[:2]
    if label.polygon is not None:
        poly = label.polygon
        x1, y1, x2, y2 = map(int, [np.min(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 0]), np.max(poly[:, 1])])
    else:
        x1, y1, x2, y2 = map(int, label.bbox)
        poly = None
    x1 = max(0, x1 - cfg.extraction.pad_px)
    y1 = max(0, y1 - cfg.extraction.pad_px)
    x2 = min(w, x2 + cfg.extraction.pad_px)
    y2 = min(h, y2 + cfg.extraction.pad_px)
    crop = image_rgba[y1:y2, x1:x2].copy()
    local_poly = None
    if poly is not None:
        local_poly = poly.copy().astype(np.float32)
        local_poly[:, 0] -= x1
        local_poly[:, 1] -= y1
    if cfg.extraction.isolation == "polygon_mask" and local_poly is not None:
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [local_poly.astype(np.int32)], 1)
        crop[..., 3] = (crop[..., 3] * mask).astype(np.uint8)
    elif cfg.extraction.isolation == "alpha_connected_component" and local_poly is None:
        alpha = crop[..., 3]
        binary = (alpha > cfg.extraction.alpha_thr).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num > 1:
            center_x = int((label.bbox[0] + label.bbox[2]) / 2) - x1
            center_y = int((label.bbox[1] + label.bbox[3]) / 2) - y1
            center_x = np.clip(center_x, 0, binary.shape[1] - 1)
            center_y = np.clip(center_y, 0, binary.shape[0] - 1)
            component_id = labels[center_y, center_x]
            if component_id == 0:
                component_id = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
            mask = (labels == component_id).astype(np.uint8)
            crop[..., 3] = (crop[..., 3] * mask).astype(np.uint8)
    bbox = label.bbox.copy().astype(np.float32)
    bbox[0] -= x1
    bbox[2] -= x1
    bbox[1] -= y1
    bbox[3] -= y1
    if local_poly is None:
        return OverlayPatch(rgba=crop, bbox=bbox, polygon=None)
    return OverlayPatch(rgba=crop, bbox=bbox, polygon=local_poly)


def _sample_transform(cfg: ClassConfig, rng: np.random.Generator) -> Tuple[float, bool, bool]:
    scale = cfg.transform.scale.sample(rng)
    hflip = rng.random() < cfg.transform.hflip_p
    vflip = rng.random() < cfg.transform.vflip_p
    return scale, hflip, vflip


def _apply_transform(
    patch: OverlayPatch, scale: float, hflip: bool, vflip: bool
) -> OverlayPatch:
    rgba = patch.rgba
    if scale != 1.0:
        h, w = rgba.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h, w = rgba.shape[:2]
    bbox = patch.bbox.copy().astype(np.float32)
    bbox = scale_boxes(bbox[None, :], scale)[0]
    poly_list = [patch.polygon] if patch.polygon is not None else [None]
    poly_list = scale_polygons(poly_list, scale)
    bbox = flip_boxes(bbox[None, :], w, h, hflip, vflip)[0]
    poly_list = flip_polygons(poly_list, w, h, hflip, vflip)
    poly = poly_list[0]
    if hflip:
        rgba = np.ascontiguousarray(rgba[:, ::-1])
    if vflip:
        rgba = np.ascontiguousarray(rgba[::-1, :])
    return OverlayPatch(rgba=rgba, bbox=bbox, polygon=poly)


def _sample_placement(
    rng: np.random.Generator,
    patch: OverlayPatch,
    base_w: int,
    base_h: int,
    cfg: ClassConfig,
    overlay_boxes: List[np.ndarray],
    allow_overlap: bool,
    max_iou: float,
) -> Optional[Tuple[int, int]]:
    h, w = patch.rgba.shape[:2]
    for _ in range(cfg.placement.max_tries):
        if cfg.placement.strategy != "uniform_in_bounds":
            raise ValueError("Only uniform_in_bounds supported")
        if cfg.placement.allow_out_of_bounds:
            x0 = int(rng.integers(-w + 1, base_w))
            y0 = int(rng.integers(-h + 1, base_h))
        else:
            if w >= base_w or h >= base_h:
                return None
            x0 = int(rng.integers(cfg.placement.margin_px, base_w - w - cfg.placement.margin_px + 1))
            y0 = int(rng.integers(cfg.placement.margin_px, base_h - h - cfg.placement.margin_px + 1))
        candidate_box = np.array([x0, y0, x0 + w, y0 + h], dtype=np.float32)
        if allow_overlap or not overlay_boxes:
            return x0, y0
        if all(compute_iou(candidate_box, box) <= max_iou for box in overlay_boxes):
            return x0, y0
    return None


def execute_plan(
    rng: np.random.Generator,
    config: SpatterAugConfig,
    asset_store: AssetStore,
    image: np.ndarray,
    instances: Instances,
    plan: List[PasteOp],
) -> Tuple[np.ndarray, Instances, Optional[dict]]:
    base_h, base_w = image.shape[:2]
    out = image.copy() if config.global_cfg.copy_base_image else image
    overlay_boxes: List[np.ndarray] = []
    coverage = None
    if config.occlusion_default.metric in {"alpha_in_bbox", "alpha_in_polygon"}:
        coverage = np.zeros((base_h, base_w), dtype=bool)
    added_boxes = []
    added_labels = []
    added_polygons: List[Optional[np.ndarray]] = []
    added_source = []
    debug_ops: List[Dict[str, object]] = []

    for op in plan:
        cls_cfg = config.classes[op.class_name]
        asset = asset_store.get_file(op.class_name, op.asset_file_id)
        image_rgba = asset_store.load_image(op.class_name, op.asset_file_id)
        if op.mode == "whole_file":
            patch = OverlayPatch(rgba=image_rgba, bbox=None, polygon=None)  # type: ignore[arg-type]
            scale, hflip, vflip = _sample_transform(cls_cfg, rng)
            if scale != 1.0 or hflip or vflip:
                patch = _apply_transform(
                    OverlayPatch(rgba=image_rgba, bbox=np.zeros(4), polygon=None),
                    scale,
                    hflip,
                    vflip,
                )
            placement = _sample_placement(
                rng,
                patch,
                base_w,
                base_h,
                cls_cfg,
                overlay_boxes,
                config.global_cfg.allow_overlay_overlay,
                config.global_cfg.overlay_overlay_max_iou,
            )
            if placement is None:
                if config.debug:
                    debug_ops.append(
                        {
                            "class_name": op.class_name,
                            "mode": op.mode,
                            "asset_file_id": op.asset_file_id,
                            "placement": None,
                            "scale": scale,
                            "hflip": hflip,
                            "vflip": vflip,
                            "skipped": "no_placement",
                        }
                    )
                continue
            x0, y0 = placement
            rgba = patch.rgba
            if cls_cfg.placement.allow_out_of_bounds:
                rgba, x0, y0 = clip_patch(rgba, x0, y0, base_w, base_h)
                if rgba.size == 0:
                    continue
            blend_roi(out, rgba, x0, y0)
            if coverage is not None:
                update_coverage(coverage, rgba[..., 3], x0, y0, config.occlusion_default.alpha_threshold)
            overlay_boxes.append(np.array([x0, y0, x0 + rgba.shape[1], y0 + rgba.shape[0]], dtype=np.float32))
            for label in asset.labels:
                bbox = label.bbox.copy().astype(np.float32)
                poly = label.polygon.copy().astype(np.float32) if label.polygon is not None else None
                if scale != 1.0:
                    bbox = scale_boxes(bbox[None, :], scale)[0]
                    poly_list = scale_polygons([poly], scale)
                    poly = poly_list[0]
                bbox = translate_boxes(bbox[None, :], x0, y0)[0]
                if poly is not None:
                    poly = translate_polygons([poly], x0, y0)[0]
                added_boxes.append(bbox)
                added_labels.append(cls_cfg.target_class_id)
                added_polygons.append(poly)
                added_source.append(1)
            if config.debug:
                debug_ops.append(
                    {
                        "class_name": op.class_name,
                        "mode": op.mode,
                        "asset_file_id": op.asset_file_id,
                        "placement": (x0, y0),
                        "scale": scale,
                        "hflip": hflip,
                        "vflip": vflip,
                        "skipped": None,
                    }
                )
        else:
            labels = asset.labels
            if not labels:
                continue
            k = min(len(labels), max(1, len(op.asset_instance_ids)))
            instance_ids = rng.choice(len(labels), size=k, replace=False)
            for idx in instance_ids:
                patch = _extract_patch(image_rgba, labels[int(idx)], cls_cfg)
                scale, hflip, vflip = _sample_transform(cls_cfg, rng)
                patch = _apply_transform(patch, scale, hflip, vflip)
                placement = _sample_placement(
                    rng,
                    patch,
                    base_w,
                    base_h,
                    cls_cfg,
                    overlay_boxes,
                    config.global_cfg.allow_overlay_overlay,
                    config.global_cfg.overlay_overlay_max_iou,
                )
                if placement is None:
                    if config.debug:
                        debug_ops.append(
                            {
                                "class_name": op.class_name,
                                "mode": op.mode,
                                "asset_file_id": op.asset_file_id,
                                "placement": None,
                                "scale": scale,
                                "hflip": hflip,
                                "vflip": vflip,
                                "skipped": "no_placement",
                            }
                        )
                    continue
                x0, y0 = placement
                rgba = patch.rgba
                poly = patch.polygon
                bbox = patch.bbox
                if cls_cfg.placement.allow_out_of_bounds:
                    rgba, x0, y0 = clip_patch(rgba, x0, y0, base_w, base_h)
                    if rgba.size == 0:
                        continue
                    poly = None
                blend_roi(out, rgba, x0, y0)
                if coverage is not None:
                    update_coverage(coverage, rgba[..., 3], x0, y0, config.occlusion_default.alpha_threshold)
                overlay_boxes.append(np.array([x0, y0, x0 + rgba.shape[1], y0 + rgba.shape[0]], dtype=np.float32))
                bbox = translate_boxes(bbox[None, :], x0, y0)[0]
                if poly is not None:
                    poly = translate_polygons([poly], x0, y0)[0]
                added_boxes.append(bbox)
                added_labels.append(cls_cfg.target_class_id)
                added_polygons.append(poly)
                added_source.append(1)
                if config.debug:
                    debug_ops.append(
                        {
                            "class_name": op.class_name,
                            "mode": op.mode,
                            "asset_file_id": op.asset_file_id,
                            "placement": (x0, y0),
                            "scale": scale,
                            "hflip": hflip,
                            "vflip": vflip,
                            "skipped": None,
                        }
                    )

    if added_boxes:
        merged_boxes = np.concatenate([instances.boxes_xyxy, np.stack(added_boxes)], axis=0)
        merged_labels = np.concatenate([instances.labels, np.array(added_labels, dtype=np.int64)], axis=0)
        merged_polygons = instances.polygons + added_polygons
        merged_source = np.concatenate([instances.source, np.array(added_source, dtype=np.int8)], axis=0)
    else:
        merged_boxes = instances.boxes_xyxy
        merged_labels = instances.labels
        merged_polygons = instances.polygons
        merged_source = instances.source

    merged = Instances(
        boxes_xyxy=merged_boxes,
        labels=merged_labels,
        polygons=merged_polygons,
        source=merged_source,
    )
    if coverage is not None:
        occ_cfg = config.occlusion_default
        base_only = occ_cfg.scope == "base_only"
        merged = drop_occluded_instances(
            merged,
            coverage,
            occ_cfg.metric,
            occ_cfg.drop_threshold,
            occ_cfg.min_box_area_px,
            base_only,
        )
    merged.boxes_xyxy = clip_boxes_xyxy(merged.boxes_xyxy, base_w, base_h)
    keep = filter_small_boxes(merged.boxes_xyxy, config.occlusion_default.min_box_area_px)
    merged = Instances(
        boxes_xyxy=merged.boxes_xyxy[keep],
        labels=merged.labels[keep],
        polygons=[p for p, k in zip(merged.polygons, keep) if k],
        source=merged.source[keep],
    )
    debug = None
    if config.debug:
        debug = {"num_overlays": len(added_boxes), "plan_ops": len(plan), "ops": debug_ops}
    return out, merged, debug
