from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .geometry import polygons_from_flat


@dataclass
class AssetLabel:
    bbox: np.ndarray
    polygon: Optional[np.ndarray]


@dataclass
class AssetFile:
    image_path: Path
    label_path: Path
    labels: List[AssetLabel]


class AssetStore:
    def __init__(self, assets_root: str, preload_images: bool = True) -> None:
        self.assets_root = Path(assets_root)
        if not self.assets_root.exists():
            raise FileNotFoundError(f"Assets root not found: {assets_root}")
        self.preload_images = preload_images
        self.class_files: Dict[str, List[AssetFile]] = {}
        self._image_cache: Dict[Tuple[str, int], np.ndarray] = {}
        self._scan()

    def _scan(self) -> None:
        for class_dir in sorted(self.assets_root.iterdir()):
            if not class_dir.is_dir():
                continue
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                continue
            files: List[AssetFile] = []
            for image_path in sorted(images_dir.glob("*.png")):
                label_path = labels_dir / f"{image_path.stem}.txt"
                image = None
                if self.preload_images or label_path.exists():
                    image = self._read_image(image_path)
                labels = self._load_labels(label_path, image, image_path)
                files.append(AssetFile(image_path=image_path, label_path=label_path, labels=labels))
                if self.preload_images and image is not None:
                    self._image_cache[(class_dir.name, len(files) - 1)] = image
            self.class_files[class_dir.name] = files

    def classes(self) -> List[str]:
        return list(self.class_files.keys())

    def file_count(self, class_name: str) -> int:
        return len(self.class_files[class_name])

    def get_file(self, class_name: str, file_id: int) -> AssetFile:
        return self.class_files[class_name][file_id]

    def load_image(self, class_name: str, file_id: int) -> np.ndarray:
        key = (class_name, file_id)
        if key in self._image_cache:
            return self._image_cache[key]
        image = self._read_image(self.get_file(class_name, file_id).image_path)
        if self.preload_images:
            self._image_cache[key] = image
        return image

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(f"Asset image must be RGBA: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    def _load_labels(
        self, label_path: Path, image: Optional[np.ndarray], image_path: Path
    ) -> List[AssetLabel]:
        if not label_path.exists():
            return []
        if image is None:
            image = self._read_image(image_path)
        h, w = image.shape[:2]
        labels: List[AssetLabel] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts[0] != "0":
                    raise ValueError("Asset labels must start with class id 0")
                coords = [float(x) for x in parts[1:]]
                if len(coords) == 4:
                    x_c, y_c, bw, bh = coords
                    x1 = (x_c - bw / 2.0) * w
                    y1 = (y_c - bh / 2.0) * h
                    x2 = (x_c + bw / 2.0) * w
                    y2 = (y_c + bh / 2.0) * h
                    bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
                    labels.append(AssetLabel(bbox=bbox, polygon=None))
                else:
                    polygon = polygons_from_flat(coords)
                    polygon[:, 0] *= w
                    polygon[:, 1] *= h
                    bbox = np.array(
                        [
                            np.min(polygon[:, 0]),
                            np.min(polygon[:, 1]),
                            np.max(polygon[:, 0]),
                            np.max(polygon[:, 1]),
                        ],
                        dtype=np.float32,
                    )
                    labels.append(AssetLabel(bbox=bbox, polygon=polygon))
        return labels
