"""Roboflow RF-DETR integration example.

RF-DETR expects DETR-style targets:
- boxes: [N, 4] in xyxy pixel coordinates
- labels: [N] int class ids
- polygons: optional list of polygons

Adjust dataset loading and trainer APIs for your RF-DETR version.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from overlayaug import AssetStore, OverlayAugmentor, OverlayAugmentedDataset


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main() -> None:
    assets = AssetStore("/path/to/assets")
    augmentor = OverlayAugmentor("/path/to/config.yaml", assets)

    # base_dataset should yield (image, target) with DETR-style keys.
    # Replace this with your Roboflow dataset builder.
    base_dataset = ...  # e.g., RoboflowDataset(root="/path/to/data")

    aug_dataset = OverlayAugmentedDataset(
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

    # Boilerplate RF-DETR training hook (adjust for your version).
    # from rfdetr import RFDETR, RFDETRTrainer
    # model = RFDETR(num_classes=base_dataset.num_classes)
    # trainer = RFDETRTrainer(model=model, train_loader=train_loader)
    # trainer.train(epochs=50)


if __name__ == "__main__":
    main()
