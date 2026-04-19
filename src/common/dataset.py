from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection

from .config import DATASET_ROOT


class TulipCocoDetection(CocoDetection):
    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        image = torchvision.transforms.functional.to_tensor(img.convert("RGB"))
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def resolve_dataset_root(dataset_root: str | None = None) -> Path:
    root = Path(dataset_root).resolve() if dataset_root else DATASET_ROOT
    required = [
        root / "train" / "_annotations.coco.json",
        root / "val" / "_annotations.coco.json",
        root / "test" / "_annotations.coco.json",
        root / "train" / "images",
        root / "val" / "images",
        root / "test" / "images",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset root is missing expected COCO files/folders:\n- " + "\n- ".join(missing)
        )
    return root


def build_dataloaders(dataset_root: str | None = None, batch_size: int = 2):
    root = resolve_dataset_root(dataset_root)

    train_dataset = TulipCocoDetection(
        root=str(root / "train" / "images"),
        annFile=str(root / "train" / "_annotations.coco.json"),
    )
    valid_dataset = TulipCocoDetection(
        root=str(root / "val" / "images"),
        annFile=str(root / "val" / "_annotations.coco.json"),
    )
    test_dataset = TulipCocoDetection(
        root=str(root / "test" / "images"),
        annFile=str(root / "test" / "_annotations.coco.json"),
    )

    pin_memory = torch.cuda.is_available()
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader
