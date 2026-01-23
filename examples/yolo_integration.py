"""Ultralytics YOLOv8 integration example.

This script shows how to wrap an Ultralytics dataset so SpatterAug can operate on
YOLO-format labels (normalized xywh). Adjust paths, class counts, and trainer
settings to match your project.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset


class UltralyticsLabelAdapterDataset:
    """Convert Ultralytics label dicts to SpatterAug's YOLO adapter format."""

    def __init__(self, base_dataset: YOLODataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        image, label = self.base_dataset[index]
        target = {
            "bboxes": label["bboxes"],
            "labels": label["cls"].astype(np.int64),
            "polygons": label.get("segments"),
        }
        return image, target


def to_ultralytics(image: np.ndarray, target: Dict[str, Any]):
    """Convert SpatterAug targets back to Ultralytics format."""
    label = {
        "bboxes": target["bboxes"],
        "cls": target["labels"],
        "segments": target.get("polygons"),
    }
    return image, label


def main() -> None:
    assets = AssetStore("/path/to/assets")
    augmentor = SpatterAugmentor("/path/to/config.yaml", assets)

    base_dataset = YOLODataset(
        data="data.yaml",
        imgsz=640,
        augment=False,
        rect=False,
        cache=False,
        prefix="",
    )
    wrapped_dataset = UltralyticsLabelAdapterDataset(base_dataset)

    aug_dataset = SpatterAugmentedDataset(
        wrapped_dataset,
        augmentor,
        adapter_in="yolo",
        adapter_out="yolo",
        post_transform=to_ultralytics,
    )

    train_loader = DataLoader(
        aug_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=base_dataset.collate_fn,
    )

    model = YOLO("yolov8n.pt")
    trainer = model.trainer(overrides={"data": "data.yaml", "epochs": 50})
    trainer.train_loader = train_loader
    trainer.train()


if __name__ == "__main__":
    main()
