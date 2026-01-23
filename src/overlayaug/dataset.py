from __future__ import annotations

from typing import Any, Callable, Optional

from .augmentor import OverlayAugmentor
from .adapters.detr import from_instances as detr_from_instances, to_instances as detr_to_instances
from .adapters.yolo import from_instances as yolo_from_instances, to_instances as yolo_to_instances


class OverlayAugmentedDataset:
    def __init__(
        self,
        base_dataset,
        augmentor: OverlayAugmentor,
        adapter_in: str,
        adapter_out: str,
        post_transform: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.adapter_in = adapter_in
        self.adapter_out = adapter_out
        self.post_transform = post_transform

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)
        self.augmentor.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        if self.adapter_in == "detr":
            instances = detr_to_instances(target)
        elif self.adapter_in == "yolo":
            instances = yolo_to_instances(target, image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Unknown adapter_in: {self.adapter_in}")

        image, instances, _ = self.augmentor(image, instances, index)

        if self.adapter_out == "detr":
            target_out = detr_from_instances(instances)
        elif self.adapter_out == "yolo":
            target_out = yolo_from_instances(instances, image.shape[1], image.shape[0])
        else:
            raise ValueError(f"Unknown adapter_out: {self.adapter_out}")

        if self.post_transform is not None:
            image, target_out = self.post_transform(image, target_out)

        return image, target_out
