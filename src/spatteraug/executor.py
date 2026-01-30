from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import warnings
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
from .torch_backend import (
    blend_roi_torch,
    blend_rois_torch,
    resize_rgba_torch,
    to_numpy_image,
    to_torch_image,
    torch_available,
)


@dataclass
class OverlayPatch:
    rgba: np.ndarray
    bbox: np.ndarray
    polygon: Optional[np.ndarray]


@dataclass
class CachedWholeFile:
    rgba: np.ndarray
    labels: List[AssetLabel]


class TransformCache:
    def __init__(self) -> None:
        self._stores: Dict[str, OrderedDict] = {}
        self._sizes: Dict[str, int] = {}
        self._torch_stores: Dict[Tuple[str, str], OrderedDict] = {}
        self._torch_sizes: Dict[Tuple[str, str], int] = {}

    def clear(self) -> None:
        self._stores.clear()
        self._sizes.clear()
        self._torch_stores.clear()
        self._torch_sizes.clear()

    def _estimate_bytes(self, value: object) -> int:
        if isinstance(value, CachedWholeFile):
            return int(value.rgba.nbytes)
        if isinstance(value, OverlayPatch):
            return int(value.rgba.nbytes)
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel() * value.element_size())  # type: ignore[no-any-return]
        if hasattr(value, "nbytes"):
            return int(value.nbytes)  # type: ignore[no-any-return]
        return 0

    def get(self, class_name: str, key: Tuple) -> Optional[object]:
        store = self._stores.get(class_name)
        if store is None:
            return None
        if key not in store:
            return None
        store.move_to_end(key)
        return store[key]

    def set(self, class_name: str, key: Tuple, value: object, max_entries: int) -> None:
        store = self._stores.setdefault(class_name, OrderedDict())
        sizes = self._sizes.setdefault(class_name, 0)
        if key in store:
            sizes -= self._estimate_bytes(store[key])
        store[key] = value
        store.move_to_end(key)
        sizes += self._estimate_bytes(value)
        self._sizes[class_name] = sizes
        while len(store) > max_entries:
            _, removed = store.popitem(last=False)
            sizes -= self._estimate_bytes(removed)
        self._sizes[class_name] = sizes

    def enforce_max_bytes(self, class_name: str, max_bytes: Optional[int]) -> None:
        if max_bytes is None:
            return
        store = self._stores.get(class_name)
        if store is None:
            return
        sizes = self._sizes.get(class_name, 0)
        while sizes > max_bytes and store:
            _, removed = store.popitem(last=False)
            sizes -= self._estimate_bytes(removed)
        self._sizes[class_name] = sizes

    def get_torch(self, class_name: str, device: str, key: Tuple) -> Optional[object]:
        store = self._torch_stores.get((class_name, device))
        if store is None:
            return None
        if key not in store:
            return None
        store.move_to_end(key)
        return store[key]

    def set_torch(self, class_name: str, device: str, key: Tuple, value: object, max_entries: int) -> None:
        store = self._torch_stores.setdefault((class_name, device), OrderedDict())
        sizes = self._torch_sizes.setdefault((class_name, device), 0)
        if key in store:
            sizes -= self._estimate_bytes(store[key])
        store[key] = value
        store.move_to_end(key)
        sizes += self._estimate_bytes(value)
        self._torch_sizes[(class_name, device)] = sizes
        while len(store) > max_entries:
            _, removed = store.popitem(last=False)
            sizes -= self._estimate_bytes(removed)
            self._torch_sizes[(class_name, device)] = sizes

    def enforce_max_bytes_torch(self, class_name: str, device: str, max_bytes: Optional[int]) -> None:
        if max_bytes is None:
            return
        store = self._torch_stores.get((class_name, device))
        if store is None:
            return
        sizes = self._torch_sizes.get((class_name, device), 0)
        while sizes > max_bytes and store:
            _, removed = store.popitem(last=False)
            sizes -= self._estimate_bytes(removed)
        self._torch_sizes[(class_name, device)] = sizes


def _cache_scale(scale: float, cfg: ClassConfig) -> float:
    rounding = cfg.transform_cache.scale_rounding
    if rounding is None:
        return scale
    return round(scale, rounding)


def _transform_labels(
    labels: List[AssetLabel],
    scale: float,
    hflip: bool,
    vflip: bool,
    patch_w: int,
    patch_h: int,
) -> List[AssetLabel]:
    if not labels:
        return []
    boxes = np.stack([label.bbox for label in labels]).astype(np.float32)
    polygons = [label.polygon.copy().astype(np.float32) if label.polygon is not None else None for label in labels]
    if scale != 1.0:
        boxes = scale_boxes(boxes, scale)
        polygons = scale_polygons(polygons, scale)
    boxes = flip_boxes(boxes, patch_w, patch_h, hflip, vflip)
    polygons = flip_polygons(polygons, patch_w, patch_h, hflip, vflip)
    return [AssetLabel(bbox=boxes[idx], polygon=polygons[idx]) for idx in range(len(labels))]


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


def _apply_transform_torch(
    rgba: "object", scale: float, hflip: bool, vflip: bool
) -> "object":
    if scale != 1.0:
        h, w = rgba.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        rgba = resize_rgba_torch(rgba, new_h, new_w)
    if hflip:
        rgba = rgba.flip(1)
    if vflip:
        rgba = rgba.flip(0)
    return rgba


def _clip_patch_torch(
    overlay_rgba: "object", x0: int, y0: int, width: int, height: int
) -> Tuple["object", int, int]:
    h, w = overlay_rgba.shape[:2]
    x1 = x0 + w
    y1 = y0 + h
    crop_x0 = max(0, -x0)
    crop_y0 = max(0, -y0)
    crop_x1 = w - max(0, x1 - width)
    crop_y1 = h - max(0, y1 - height)
    if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
        return overlay_rgba[:0, :0], 0, 0
    clipped = overlay_rgba[crop_y0:crop_y1, crop_x0:crop_x1]
    return clipped, x0 + crop_x0, y0 + crop_y0


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
    transform_cache: Optional[TransformCache] = None,
) -> Tuple[np.ndarray, Instances, Optional[dict]]:
    base_h, base_w = image.shape[:2]
    out = image.copy() if config.global_cfg.copy_base_image else image
    use_torch = config.global_cfg.backend == "torch"
    torch_device = config.global_cfg.torch_device
    out_torch = None
    if use_torch:
        if not torch_available():
            raise RuntimeError("Torch backend requested but torch is not installed.")
        out_torch = to_torch_image(out, torch_device)
    blend_overlays: List[object] = []
    blend_xs: List[int] = []
    blend_ys: List[int] = []
    overlay_boxes: List[np.ndarray] = []
    coverage = None
    if config.occlusion_default.metric in {"alpha_in_bbox", "alpha_in_polygon"}:
        coverage = np.zeros((base_h, base_w), dtype=bool)
    added_boxes = []
    added_labels = []
    added_polygons: List[Optional[np.ndarray]] = []
    added_source = []

    warned_classes = set()
    for op in plan:
        cls_cfg = config.classes[op.class_name]
        if (
            cls_cfg.transform_cache.enabled
            and cls_cfg.transform_cache.scale_rounding is None
            and op.class_name not in warned_classes
        ):
            warnings.warn(
                "Transform cache enabled without scale_rounding; cache hits may be rare.",
                RuntimeWarning,
                stacklevel=2,
            )
            warned_classes.add(op.class_name)
        asset = asset_store.get_file(op.class_name, op.asset_file_id)
        image_rgba = asset_store.load_image(op.class_name, op.asset_file_id)
        if op.mode == "whole_file":
            patch = OverlayPatch(rgba=image_rgba, bbox=None, polygon=None)  # type: ignore[arg-type]
            scale, hflip, vflip = _sample_transform(cls_cfg, rng)
            if cls_cfg.transform_cache.enabled and cls_cfg.transform_cache.scale_rounding is not None:
                scale = _cache_scale(scale, cls_cfg)
            overlay_torch = None
            cached_labels = None
            if cls_cfg.transform_cache.enabled and transform_cache is not None:
                cache_key = ("whole", op.asset_file_id, scale, hflip, vflip)
                cached = transform_cache.get(op.class_name, cache_key)
                if isinstance(cached, CachedWholeFile):
                    patch = OverlayPatch(rgba=cached.rgba, bbox=None, polygon=None)  # type: ignore[arg-type]
                    cached_labels = cached.labels
                if use_torch and out_torch is not None:
                    overlay_torch = transform_cache.get_torch(op.class_name, torch_device, cache_key)
            if cached_labels is None and (scale != 1.0 or hflip or vflip):
                if use_torch and out_torch is not None:
                    overlay_torch = asset_store.load_image_torch(op.class_name, op.asset_file_id, torch_device)
                    overlay_torch = _apply_transform_torch(overlay_torch, scale, hflip, vflip)
                    patch = OverlayPatch(rgba=to_numpy_image(overlay_torch), bbox=np.zeros(4), polygon=None)
                else:
                    patch = _apply_transform(
                        OverlayPatch(rgba=image_rgba, bbox=np.zeros(4), polygon=None),
                        scale,
                        hflip,
                        vflip,
                    )
            if cached_labels is None:
                patch_h, patch_w = patch.rgba.shape[:2]
                cached_labels = _transform_labels(asset.labels, scale, hflip, vflip, patch_w, patch_h)
                if cls_cfg.transform_cache.enabled and transform_cache is not None:
                    cache_key = ("whole", op.asset_file_id, scale, hflip, vflip)
                    transform_cache.set(
                        op.class_name,
                        cache_key,
                        CachedWholeFile(rgba=patch.rgba, labels=cached_labels),
                        cls_cfg.transform_cache.max_entries,
                    )
                    transform_cache.enforce_max_bytes(op.class_name, cls_cfg.transform_cache.max_bytes)
                    if overlay_torch is not None:
                        transform_cache.set_torch(
                            op.class_name,
                            torch_device,
                            cache_key,
                            overlay_torch,
                            cls_cfg.transform_cache.max_entries,
                        )
                        transform_cache.enforce_max_bytes_torch(
                            op.class_name,
                            torch_device,
                            cls_cfg.transform_cache.max_bytes,
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
                continue
            x0, y0 = placement
            rgba = patch.rgba
            crop_dx = 0
            crop_dy = 0
            if cls_cfg.placement.allow_out_of_bounds:
                old_x0, old_y0 = x0, y0
                if use_torch and out_torch is not None:
                    if overlay_torch is None:
                        overlay_torch = to_torch_image(rgba, torch_device)
                    overlay_torch, x0, y0 = _clip_patch_torch(overlay_torch, x0, y0, base_w, base_h)
                    rgba = to_numpy_image(overlay_torch)
                else:
                    rgba, x0, y0 = clip_patch(rgba, x0, y0, base_w, base_h)
                crop_dx = x0 - old_x0
                crop_dy = y0 - old_y0
                if rgba.size == 0:
                    continue
            if use_torch and out_torch is not None:
                if overlay_torch is None:
                    if scale == 1.0 and not hflip and not vflip and rgba is image_rgba:
                        overlay_torch = asset_store.load_image_torch(op.class_name, op.asset_file_id, torch_device)
                    else:
                        overlay_torch = to_torch_image(rgba, torch_device)
                blend_overlays.append(overlay_torch)
                blend_xs.append(x0)
                blend_ys.append(y0)
            else:
                blend_roi(out, rgba, x0, y0)
            if coverage is not None:
                update_coverage(coverage, rgba[..., 3], x0, y0, config.occlusion_default.alpha_threshold)
            overlay_boxes.append(np.array([x0, y0, x0 + rgba.shape[1], y0 + rgba.shape[0]], dtype=np.float32))
            for label in cached_labels:
                bbox = label.bbox.copy().astype(np.float32)
                poly = label.polygon.copy().astype(np.float32) if label.polygon is not None else None
                if crop_dx != 0 or crop_dy != 0:
                    bbox = translate_boxes(bbox[None, :], -crop_dx, -crop_dy)[0]
                    if poly is not None:
                        poly = translate_polygons([poly], -crop_dx, -crop_dy)[0]
                bbox = translate_boxes(bbox[None, :], x0, y0)[0]
                if poly is not None:
                    poly = translate_polygons([poly], x0, y0)[0]
                added_boxes.append(bbox)
                added_labels.append(cls_cfg.target_class_id)
                added_polygons.append(poly)
                added_source.append(1)
        else:
            labels = asset.labels
            if not labels:
                continue
            k = min(len(labels), len(op.asset_instance_ids))
            if k <= 0:
                continue
            instance_ids = rng.choice(len(labels), size=k, replace=False)
            for idx in instance_ids:
                scale, hflip, vflip = _sample_transform(cls_cfg, rng)
                if cls_cfg.transform_cache.enabled and cls_cfg.transform_cache.scale_rounding is not None:
                    scale = _cache_scale(scale, cls_cfg)
                overlay_torch = None
                cached_patch = None
                if cls_cfg.transform_cache.enabled and transform_cache is not None:
                    cache_key = (
                        "instance",
                        op.asset_file_id,
                        int(idx),
                        scale,
                        hflip,
                        vflip,
                        cls_cfg.extraction.isolation,
                        cls_cfg.extraction.pad_px,
                        cls_cfg.extraction.alpha_thr,
                    )
                    cached = transform_cache.get(op.class_name, cache_key)
                    if isinstance(cached, OverlayPatch):
                        cached_patch = cached
                    if use_torch and out_torch is not None:
                        overlay_torch = transform_cache.get_torch(op.class_name, torch_device, cache_key)
                if cached_patch is None:
                    patch = _extract_patch(image_rgba, labels[int(idx)], cls_cfg)
                    if use_torch and out_torch is not None and (scale != 1.0 or hflip or vflip):
                        overlay_torch = to_torch_image(patch.rgba, torch_device)
                        overlay_torch = _apply_transform_torch(overlay_torch, scale, hflip, vflip)
                        rgba = to_numpy_image(overlay_torch)
                        patch_h, patch_w = rgba.shape[:2]
                        bbox = patch.bbox.copy().astype(np.float32)
                        poly = patch.polygon.copy().astype(np.float32) if patch.polygon is not None else None
                        if scale != 1.0:
                            bbox = scale_boxes(bbox[None, :], scale)[0]
                            poly_list = scale_polygons([poly], scale)
                            poly = poly_list[0]
                        bbox = flip_boxes(bbox[None, :], patch_w, patch_h, hflip, vflip)[0]
                        poly = flip_polygons([poly], patch_w, patch_h, hflip, vflip)[0]
                        patch = OverlayPatch(rgba=rgba, bbox=bbox, polygon=poly)
                    else:
                        patch = _apply_transform(patch, scale, hflip, vflip)
                    cached_patch = patch
                    if cls_cfg.transform_cache.enabled and transform_cache is not None:
                        cache_key = (
                            "instance",
                            op.asset_file_id,
                            int(idx),
                            scale,
                            hflip,
                            vflip,
                            cls_cfg.extraction.isolation,
                            cls_cfg.extraction.pad_px,
                            cls_cfg.extraction.alpha_thr,
                        )
                        transform_cache.set(
                            op.class_name,
                            cache_key,
                            cached_patch,
                            cls_cfg.transform_cache.max_entries,
                        )
                        transform_cache.enforce_max_bytes(op.class_name, cls_cfg.transform_cache.max_bytes)
                        if overlay_torch is not None:
                            transform_cache.set_torch(
                                op.class_name,
                                torch_device,
                                cache_key,
                                overlay_torch,
                                cls_cfg.transform_cache.max_entries,
                            )
                            transform_cache.enforce_max_bytes_torch(
                                op.class_name,
                                torch_device,
                                cls_cfg.transform_cache.max_bytes,
                            )
                patch = cached_patch
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
                    continue
                x0, y0 = placement
                rgba = patch.rgba
                poly = patch.polygon
                bbox = patch.bbox
                crop_dx = 0
                crop_dy = 0
                if cls_cfg.placement.allow_out_of_bounds:
                    old_x0, old_y0 = x0, y0
                    if use_torch and out_torch is not None:
                        if overlay_torch is None:
                            overlay_torch = to_torch_image(rgba, torch_device)
                        overlay_torch, x0, y0 = _clip_patch_torch(overlay_torch, x0, y0, base_w, base_h)
                        rgba = to_numpy_image(overlay_torch)
                    else:
                        rgba, x0, y0 = clip_patch(rgba, x0, y0, base_w, base_h)
                    crop_dx = x0 - old_x0
                    crop_dy = y0 - old_y0
                    if rgba.size == 0:
                        continue
                    poly = None
                if crop_dx != 0 or crop_dy != 0:
                    bbox = translate_boxes(bbox[None, :], -crop_dx, -crop_dy)[0]
                if use_torch and out_torch is not None:
                    if overlay_torch is None:
                        overlay_torch = to_torch_image(rgba, torch_device)
                    blend_overlays.append(overlay_torch)
                    blend_xs.append(x0)
                    blend_ys.append(y0)
                else:
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
        debug = {"num_overlays": len(added_boxes), "plan_ops": len(plan)}
    if use_torch and out_torch is not None:
        if blend_overlays:
            blend_rois_torch(out_torch, blend_overlays, blend_xs, blend_ys)
        out_np = to_numpy_image(out_torch)
        if not config.global_cfg.copy_base_image:
            np.copyto(image, out_np)
            out = image
        else:
            out = out_np
    return out, merged, debug
