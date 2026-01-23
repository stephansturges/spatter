from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class UniformInt:
    low: int
    high: int

    def sample(self, rng):
        return int(rng.integers(self.low, self.high + 1))


@dataclass
class UniformFloat:
    min: float
    max: float

    def sample(self, rng):
        return float(rng.uniform(self.min, self.max))


@dataclass
class ApplyConfig:
    p: float
    conditions: Dict[str, Optional[int]] = field(default_factory=dict)


@dataclass
class ModeConfig:
    type: str
    mixed_weights: Optional[Dict[str, float]] = None


@dataclass
class IntensityConfig:
    n_files: UniformInt
    k_instances_per_file: Optional[UniformInt] = None


@dataclass
class ExtractionConfig:
    isolation: str = "alpha_connected_component"
    pad_px: int = 0
    alpha_thr: int = 10


@dataclass
class TransformConfig:
    scale: UniformFloat
    hflip_p: float = 0.0
    vflip_p: float = 0.0


@dataclass
class PlacementConfig:
    strategy: str = "uniform_in_bounds"
    allow_out_of_bounds: bool = False
    max_tries: int = 10
    margin_px: int = 0


@dataclass
class OcclusionConfig:
    scope: str = "base_only"
    metric: str = "alpha_in_bbox"
    alpha_threshold: int = 10
    drop_threshold: float = 0.7
    min_box_area_px: int = 16


@dataclass
class GlobalConfig:
    strategy: str = "independent"
    max_total_pastes: int = 100
    max_total_instances: int = 300
    allow_overlay_overlay: bool = False
    overlay_overlay_max_iou: float = 0.01
    copy_base_image: bool = True


@dataclass
class ClassConfig:
    name: str
    assets_dir: str
    target_class_id: int
    apply: ApplyConfig
    mode: ModeConfig
    intensity: IntensityConfig
    extraction: ExtractionConfig
    transform: TransformConfig
    placement: PlacementConfig
    occlusion_override: Optional[Dict[str, Any]] = None


@dataclass
class SpatterAugConfig:
    seed: int = 12345
    global_cfg: GlobalConfig = field(default_factory=GlobalConfig)
    occlusion_default: OcclusionConfig = field(default_factory=OcclusionConfig)
    classes: Dict[str, ClassConfig] = field(default_factory=dict)
    debug: bool = False


DIST_MAP = {
    "uniform_int": UniformInt,
    "uniform_float": UniformFloat,
}

SIMPLE_PRESET = "simple_overlay"


def _normalize_class_config(name: str, cls_cfg: Any, preset: Optional[str]) -> Dict[str, Any]:
    if isinstance(cls_cfg, int):
        cls_cfg = {"target_class_id": cls_cfg}
    if not isinstance(cls_cfg, dict):
        raise ValueError(f"Class config for {name} must be dict or int")
    normalized = dict(cls_cfg)
    if "assets_dir" not in normalized:
        normalized["assets_dir"] = name
    if preset == SIMPLE_PRESET and "apply" not in normalized:
        normalized["apply"] = {"p": 1.0}
    return normalized


def _parse_dist(cfg: Dict[str, Any]):
    dist_type = cfg.get("dist")
    if dist_type not in DIST_MAP:
        raise ValueError(f"Unsupported dist type: {dist_type}")
    cls = DIST_MAP[dist_type]
    params = {k: v for k, v in cfg.items() if k != "dist"}
    return cls(**params)


def _validate_config(config: Dict[str, Any], classes_cfg: Dict[str, ClassConfig]) -> None:
    errors: List[str] = []
    if "classes" not in config:
        errors.append("Missing required top-level key: classes")
    if not classes_cfg:
        errors.append("No classes configured")
    for name, cls_cfg in classes_cfg.items():
        if cls_cfg.assets_dir is None:
            errors.append(f"Class {name} missing assets_dir")
        if cls_cfg.target_class_id is None:
            errors.append(f"Class {name} missing target_class_id")
        if cls_cfg.mode.type == "mixed" and not cls_cfg.mode.mixed_weights:
            errors.append(f"Class {name} mode=mixed requires mixed_weights")
    if errors:
        raise ValueError("Invalid config:\n- " + "\n- ".join(errors))


def load_config(config: Dict[str, Any] | str) -> SpatterAugConfig:
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be dict or path")

    preset = config.get("preset")
    seed = config.get("seed", 12345)
    global_cfg = GlobalConfig(**config.get("global", {}))
    occlusion_default = OcclusionConfig(**config.get("occlusion_default", {}))
    debug = bool(config.get("debug", False))

    classes_cfg = {}
    classes = config.get("classes", {})
    if not classes:
        raise ValueError("Config must include classes")

    for name, cls_cfg in classes.items():
        cls_cfg = _normalize_class_config(name, cls_cfg, preset)
        if "target_class_id" not in cls_cfg:
            raise ValueError(f"Class {name} missing target_class_id")
        apply = ApplyConfig(**cls_cfg.get("apply", {"p": 0.0}))
        mode = ModeConfig(**cls_cfg.get("mode", {"type": "whole_file"}))
        intensity_cfg = cls_cfg.get("intensity", {})
        n_files = _parse_dist(intensity_cfg.get("n_files", {"dist": "uniform_int", "low": 1, "high": 1}))
        k_instances = None
        if "k_instances_per_file" in intensity_cfg:
            k_instances = _parse_dist(intensity_cfg["k_instances_per_file"])
        intensity = IntensityConfig(n_files=n_files, k_instances_per_file=k_instances)
        extraction = ExtractionConfig(**cls_cfg.get("extraction", {}))
        transform_cfg = cls_cfg.get("transform", {})
        scale = _parse_dist(transform_cfg.get("scale", {"dist": "uniform_float", "min": 1.0, "max": 1.0}))
        transform = TransformConfig(scale=scale, hflip_p=transform_cfg.get("hflip_p", 0.0), vflip_p=transform_cfg.get("vflip_p", 0.0))
        placement = PlacementConfig(**cls_cfg.get("placement", {}))
        class_config = ClassConfig(
            name=name,
            assets_dir=cls_cfg["assets_dir"],
            target_class_id=int(cls_cfg["target_class_id"]),
            apply=apply,
            mode=mode,
            intensity=intensity,
            extraction=extraction,
            transform=transform,
            placement=placement,
            occlusion_override=cls_cfg.get("occlusion_override"),
        )
        classes_cfg[name] = class_config

    _validate_config(config, classes_cfg)
    return SpatterAugConfig(
        seed=seed,
        global_cfg=global_cfg,
        occlusion_default=occlusion_default,
        classes=classes_cfg,
        debug=debug,
    )
