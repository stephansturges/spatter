from __future__ import annotations

from typing import Any, Callable, Optional

from .assets import AssetStore
from .augmentor import SpatterAugmentor
from .dataset import SpatterAugmentedDataset


def wrap_dataset(
    base_dataset: Any,
    assets_root: str,
    config: dict | str,
    *,
    adapter_in: str | Callable[[Any, int, int], Any] = "auto",
    adapter_out: str | Callable[[Any, int, int], Any] = "auto",
    post_transform: Optional[Callable[[Any, Any], Any]] = None,
    preload_images: bool = True,
    enabled: bool = True,
) -> SpatterAugmentedDataset:
    asset_store = AssetStore(assets_root, preload_images=preload_images)
    augmentor = SpatterAugmentor(config, asset_store, enabled=enabled)
    return SpatterAugmentedDataset(
        base_dataset,
        augmentor,
        adapter_in=adapter_in,
        adapter_out=adapter_out,
        post_transform=post_transform,
    )


def wrap_detr_dataset(
    base_dataset: Any,
    assets_root: str,
    config: dict | str,
    *,
    post_transform: Optional[Callable[[Any, Any], Any]] = None,
    preload_images: bool = True,
    enabled: bool = True,
) -> SpatterAugmentedDataset:
    return wrap_dataset(
        base_dataset,
        assets_root,
        config,
        adapter_in="detr",
        adapter_out="detr",
        post_transform=post_transform,
        preload_images=preload_images,
        enabled=enabled,
    )


def wrap_yolo_dataset(
    base_dataset: Any,
    assets_root: str,
    config: dict | str,
    *,
    post_transform: Optional[Callable[[Any, Any], Any]] = None,
    preload_images: bool = True,
    enabled: bool = True,
) -> SpatterAugmentedDataset:
    return wrap_dataset(
        base_dataset,
        assets_root,
        config,
        adapter_in="yolo",
        adapter_out="yolo",
        post_transform=post_transform,
        preload_images=preload_images,
        enabled=enabled,
    )
