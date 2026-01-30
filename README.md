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
  backend: cpu
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
    transform_cache:
      enabled: false
      max_entries: 128
      scale_rounding: null
      max_bytes: null
    placement:
      strategy: uniform_in_bounds
```

- `target_class_id` is the class id in your **training dataset** (not the asset labels).
- `assets_dir` is the subdirectory under the assets root.
- `apply.p` controls the probability of pasting that class on a given sample.
- `transform_cache` optionally caches transformed overlays to skip repeated scale/flip work.
  - `scale_rounding` (when set) quantizes sampled scales to improve cache hit rates.
  - `max_bytes` (when set) evicts cached overlays based on approximate memory usage.

### Backends

SpatterAug supports a Torch backend for GPU blending when `backend: torch` is set under `global`.
This keeps the base image and overlays in torch tensors while compositing. Torch is an optional
dependency; if it is missing and the torch backend is requested, a runtime error is raised. Use
`torch_device` under `global` to select a device (defaults to `cuda`).

**Performance profile:** the Torch backend accelerates per-overlay blend plus scale/flip transforms,
but the pipeline still executes overlays serially in Python. GPU blending helps most when you paste
many large overlays, while CPU-only workloads benefit from `transform_cache` with `scale_rounding`
to improve cache hit rates.

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
`yolo` adapter format and back. See the full example in `examples/yolo_integration.py`. The goal is
to keep the rest of the Ultralytics training loop untouched while swapping the dataset for the
SpatterAug wrapper.

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

### Step-by-step

1. **Create the augmentor.** Point to your assets directory (PNG masks/textures) and a config file.
2. **Wrap the Ultralytics dataset** so labels match SpatterAug's YOLO adapter schema.
3. **Wrap the adapted dataset** with `SpatterAugmentedDataset`.
4. **Swap the Ultralytics train loader** (or use a custom loop) so training pulls from SpatterAug.

The adapter expects normalized `xywh` boxes and integer class labels. Ultralytics already stores
`label["bboxes"]` in normalized `xywh`, and `label["cls"]` as class ids, so the adapter is mostly a
field rename.

```python
from ultralytics.data.dataset import YOLODataset
from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset

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

class UltralyticsLabelAdapterDataset:
    def __init__(self, base_dataset: YOLODataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        target = {
            "bboxes": label["bboxes"],  # normalized xywh
            "labels": label["cls"].astype("int64"),
            "polygons": label.get("segments"),
        }
        return image, target

def to_ultralytics(image, target):
    return image, {
        "bboxes": target["bboxes"],
        "cls": target["labels"],
        "segments": target.get("polygons"),
    }

wrapped_dataset = UltralyticsLabelAdapterDataset(base_dataset)

aug_dataset = SpatterAugmentedDataset(
    wrapped_dataset,
    augmentor,
    adapter_in="yolo",
    adapter_out="yolo",
    post_transform=to_ultralytics,
)
```

### Minimal trainer wiring in a YOLOv8 repo

If you are inside the Ultralytics repo (or a project using `YOLO(...)`), override the trainer's
`train_loader` so everything else (model creation, loss, hooks) stays the same. This is intentionally
minimal so you can drop it into a script or notebook:

```python
from torch.utils.data import DataLoader
from ultralytics import YOLO

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
```

**Checklist when wiring YOLOv8:**
- Confirm `label["bboxes"]` are normalized `xywh` (Ultralytics default).
- Ensure `label["cls"]` is `int64`.
- Keep Ultralytics' `collate_fn` so batch formatting stays compatible.

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

### Step-by-step

1. **Create your RF-DETR dataset** so it yields `(image, target)` with:
   - `target["boxes"]`: `[N, 4]` in `xyxy` pixel coordinates.
   - `target["labels"]`: `[N]` integer class ids.
   - Optional `target["polygons"]`: list of polygons if you have segmentation.
2. **Wrap with `SpatterAugmentedDataset`** using `adapter_in="detr"` and `adapter_out="detr"`.
3. **Use a DETR-style collate function** that returns `list[images], list[targets]`.
4. **Feed that loader into RF-DETR's trainer** (or your custom loop).

```python
from torch.utils.data import DataLoader
from spatteraug import AssetStore, SpatterAugmentor, SpatterAugmentedDataset

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

assets = AssetStore("/path/to/assets")
augmentor = SpatterAugmentor("/path/to/config.yaml", assets)

base_dataset = ...  # your RF-DETR dataset

aug_dataset = SpatterAugmentedDataset(
    base_dataset,
    augmentor,
    adapter_in="detr",
    adapter_out="detr",
)

train_loader = DataLoader(
    aug_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)
```

### Minimal wiring in an RF-DETR repo

RF-DETR training code varies by release, but the core idea is to pass the loader you created above
into the trainer or training loop. If your RF-DETR entrypoint expects a dataset object, you can hand
it `aug_dataset` directly (since it implements `__len__` and `__getitem__`).

```python
# Pseudocode: update for your RF-DETR version.
# from rfdetr import RFDETR, RFDETRTrainer

# model = RFDETR(num_classes=base_dataset.num_classes)
# trainer = RFDETRTrainer(model=model, train_loader=train_loader)
# trainer.train(epochs=50)
```

**Checklist when wiring RF-DETR:**
- Keep `boxes` in absolute pixel `xyxy` coordinates.
- Ensure `labels` are integer class ids.
- Use a collate function that returns lists (DETR-style).

## Examples

- `examples/yolo_integration.py`: Ultralytics YOLOv8 integration (dataset wrapper + adapter mapping).
- `examples/detr_integration.py`: RF-DETR integration with DETR-style targets.
- `examples/visualize_samples.py`: Visualize augmented samples.

## Testing

```bash
pytest
```
