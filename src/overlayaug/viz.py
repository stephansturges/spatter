from __future__ import annotations

from typing import List, Optional

import numpy as np
import cv2

from .geometry import Instances


def render(image: np.ndarray, instances: Instances, class_names: Optional[List[str]] = None) -> np.ndarray:
    canvas = image.copy()
    for box, label in zip(instances.boxes_xyxy, instances.labels):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0) if label < 1 else (255, 0, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = str(label)
        if class_names and label < len(class_names):
            name = class_names[int(label)]
        cv2.putText(canvas, name, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return canvas
