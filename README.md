# SpatterAug

SpatterAug is an occlusion-aware overlay augmentation library for PyTorch detection training. It provides
reproducible overlay placement, asset management, compositing, and adapter utilities so you can augment
existing datasets without rewriting your training loop.

## Installation

```bash
pip install -e .
```

SpatterAug uses `numpy`, `opencv-python`, and `pyyaml`. Make sure OpenCV has PNG alpha support
(`opencv-python` includes it by default).

## Asset layout

Overlay assets are organized per class. Each class directory must include `images/` and `labels/` with
matching filenames. Asset labels are in YOLO format and **must use class id `0`** because the class
mapping is driven by your overlay configuration.

```
assets/
  helmet/
    images/
      0001.png
    labels/
      0001.txt
  vest/
    images/
      0002.png
    labels/
      0002.txt
```

Label files can be either:
- **Bounding box**: `0 x_center y_center width height` (normalized)
- **Polygon**: `0 x1 y1 x2 y2 ...` (normalized)

## Configuration

SpatterAug uses a YAML/JSON config that describes which asset classes to paste, how often, and how to
place them. A minimal YAML file looks like this:

```yaml
seed: 12345
global:
  strategy: independent
  max_total_pastes: 100
occlusion_default:
  drop_threshold: 0.7
classes:
  helmet:
    assets_dir: helmet
    target_class_id: 2
    apply:
      p: 0.6
    mode:
      type: whole_file
    intensity:
      n_files:
        dist: uniform_int
        low: 1
        high: 2
    extraction:
      isolation: alpha_connected_component
    transform:
      scale:
        dist: uniform_float
        min: 0.6
        max: 1.2
      hflip_p: 0.5
    placement:
      strategy: uniform_in_bounds
```

- `target_class_id` is the class id in your **training dataset** (not the asset labels).
- `assets_dir` is the subdirectory under the assets root.
- `apply.p` controls the probability of pasting that class on a given sample.

## Core API

### AssetStore

```python
from spatteraug import AssetStore

assets = AssetStore("/path/to/assets", preload_images=True)
```

### SpatterAugmentor

```python
import numpy as np
from spatteraug import AssetStore, SpatterAugmentor

assets = AssetStore("/path/to/assets")
augmentor = SpatterAugmentor("/path/to/config.yaml", assets)

image = np.zeros((720, 1280, 3), dtype=np.uint8)
# DETR-style target (boxes in xyxy pixels)
target = {
    "boxes": np.array([[100, 100, 200, 200]], dtype=np.float32),
    "labels": np.array([1], dtype=np.int64),
}

aug_image, aug_target, debug = augmentor(image, target, index=0)
```

### SpatterAugmentedDataset

Wrap an existing dataset that yields `(image, target)` pairs. Use the built-in adapters to convert
between the dataset label format and SpatterAug's internal `Instances`.

```python
from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset

assets = AssetStore("/path/to/assets")
augmentor = SpatterAugmentor("/path/to/config.yaml", assets)

aug_dataset = SpatterAugmentedDataset(
    base_dataset,
    augmentor,
    adapter_in="detr",
    adapter_out="detr",
)
```

Supported adapters:
- `detr`: `{"boxes": xyxy_px, "labels": class_ids}`
- `yolo`: `{"bboxes": xywh_norm, "labels": class_ids, "polygons": optional}`

## Ultralytics YOLOv8 integration

Ultralytics datasets expose YOLO-style labels. The example below maps Ultralytics label dicts to the
`yolo` adapter format and back. See the full example in `examples/yolo_integration.py`.

```python
from ultralytics.data.dataset import YOLODataset
from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset

base_dataset = YOLODataset(
    data="data.yaml",
    imgsz=640,
    augment=False,
    rect=False,
    cache=False,
    prefix="",
)
```

You can then wrap `base_dataset` with `SpatterAugmentedDataset` and use it inside a custom training
loop or by overriding the Ultralytics trainer dataloader. The adapter expects normalized `xywh` boxes
and integer class labels.

## Roboflow RF-DETR integration

Roboflow's RF-DETR uses DETR-style targets with `boxes` in `xyxy` pixel coordinates. Wrap the dataset
with the `detr` adapters so the output stays compatible with RF-DETR. A boilerplate example is in
`examples/detr_integration.py`.

```python
from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset

assets = AssetStore("/path/to/assets")
augmentor = SpatterAugmentor("/path/to/config.yaml", assets)

aug_dataset = SpatterAugmentedDataset(
    rfdetr_dataset,
    augmentor,
    adapter_in="detr",
    adapter_out="detr",
)
```

## Examples

- `examples/yolo_integration.py`: Ultralytics YOLOv8 integration (dataset wrapper + adapter mapping).
- `examples/detr_integration.py`: RF-DETR integration with DETR-style targets.
- `examples/visualize_samples.py`: Visualize augmented samples.

## Testing

```bash
pytest
```
