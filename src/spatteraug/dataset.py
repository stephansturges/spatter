from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .augmentor import SpatterAugmentor
from .adapters.detr import from_instances as detr_from_instances, to_instances as detr_to_instances
from .adapters.yolo import from_instances as yolo_from_instances, to_instances as yolo_to_instances
from .geometry import Instances

AdapterPair = Tuple[
    Callable[[Any, int, int], Instances],
    Callable[[Instances, int, int], Any],
]


def _to_instances_detr(target: Any, width: int, height: int) -> Instances:
    return detr_to_instances(target)


def _from_instances_detr(instances: Instances, width: int, height: int) -> Any:
    return detr_from_instances(instances)


def _to_instances_yolo(target: Any, width: int, height: int) -> Instances:
    return yolo_to_instances(target, width, height)


def _from_instances_yolo(instances: Instances, width: int, height: int) -> Any:
    return yolo_from_instances(instances, width, height)


ADAPTERS: Dict[str, AdapterPair] = {
    "detr": (_to_instances_detr, _from_instances_detr),
    "yolo": (_to_instances_yolo, _from_instances_yolo),
}


def register_adapter(name: str, to_instances: Callable[[Any, int, int], Instances], from_instances: Callable[[Instances, int, int], Any]) -> None:
    ADAPTERS[name] = (to_instances, from_instances)


class SpatterAugmentedDataset:
    def __init__(
        self,
        base_dataset,
        augmentor: SpatterAugmentor,
        adapter_in: str | Callable[[Any, int, int], Any],
        adapter_out: str | Callable[[Any, int, int], Any],
        post_transform: Optional[Callable[[Any, Any], Any]] = None,
        prefer_adapter: Optional[str] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.adapter_in = adapter_in
        self.adapter_out = adapter_out
        self.post_transform = post_transform
        self.prefer_adapter = prefer_adapter

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)
        self.augmentor.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target = self.base_dataset[index]
        adapter_in = self.adapter_in
        if callable(adapter_in):
            instances = adapter_in(target, image.shape[1], image.shape[0])
        else:
            if adapter_in == "auto":
                adapter_in = _infer_adapter(target, self.prefer_adapter)
            if adapter_in in ADAPTERS:
                instances = ADAPTERS[adapter_in][0](target, image.shape[1], image.shape[0])
            else:
                raise ValueError(f"Unknown adapter_in: {adapter_in}")

        image, instances, _ = self.augmentor(image, instances, index)

        adapter_out = self.adapter_out
        if callable(adapter_out):
            target_out = adapter_out(instances, image.shape[1], image.shape[0])
        else:
            if adapter_out == "auto":
                adapter_out = _infer_adapter(target, self.prefer_adapter)
            if adapter_out in ADAPTERS:
                target_out = ADAPTERS[adapter_out][1](instances, image.shape[1], image.shape[0])
            else:
                raise ValueError(f"Unknown adapter_out: {adapter_out}")

        if self.post_transform is not None:
            image, target_out = self.post_transform(image, target_out)

        return image, target_out


def _infer_adapter(target: Any, prefer: Optional[str]) -> str:
    if isinstance(target, dict):
        has_boxes = "boxes" in target and "labels" in target
        has_bboxes = "bboxes" in target and "labels" in target
        if has_boxes and has_bboxes:
            if prefer in {"detr", "yolo"}:
                return prefer
            raise ValueError("Ambiguous adapter inference: target has both boxes and bboxes")
        if has_boxes:
            return "detr"
        if has_bboxes:
            return "yolo"
    raise ValueError("Unable to infer adapter type from target")
