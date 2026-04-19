import argparse
import copy
import os
from pathlib import Path

import torch
import torchvision
from roboflow import Roboflow
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou


PROJECT_DIR = Path(__file__).resolve().parent
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "3XHE68nOsu9vv33Z7uVk")


class TulipCocoDetection(CocoDetection):
    def __init__(self, *args, augment=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment = False

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)
        image_id = self.ids[idx]
        width, _ = img.size

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

        if self.augment and torch.rand(1).item() < 0.5:
            img = torchvision.transforms.functional.hflip(img)
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes

        img = torchvision.transforms.functional.to_tensor(img.convert("RGB"))
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def stabilize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def ensure_dataset(project_dir: Path) -> Path:
    dataset_dir = project_dir
    annotation_check = dataset_dir / "train" / "_annotations.coco.json"
    if annotation_check.exists():
        print(f"Using existing dataset in: {dataset_dir}")
        return dataset_dir

    print("Roboflow export not found locally. Downloading COCO export...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("tasnims-workspace-o6fvc").project("tulip-project")
    version = project.version(7)
    dataset = version.download("coco", location=str(project_dir), overwrite=True)
    dataset_dir = Path(dataset.location)
    print(f"Dataset downloaded to: {dataset_dir}")
    return dataset_dir


def build_model(device: torch.device):
    try:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception as exc:
        print(f"Falling back to randomly initialized weights: {exc}")
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        2,
    )
    model.to(device)
    return model


def evaluate_map50(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            preds = []
            gts = []

            for output, target in zip(outputs, targets):
                preds.append(
                    {
                        "boxes": output["boxes"].detach().cpu(),
                        "scores": output["scores"].detach().cpu(),
                        "labels": output["labels"].detach().cpu(),
                    }
                )
                gts.append(
                    {
                        "boxes": target["boxes"].detach().cpu(),
                        "labels": target["labels"].detach().cpu(),
                    }
                )

            metric.update(preds, gts)

    return metric.compute()


def compute_prf1(model, dataloader, device, score_threshold=0.5, iou_threshold=0.5):
    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                gt_boxes = target["boxes"].detach().cpu()

                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]

                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue

                if len(gt_boxes) == 0:
                    false_positives += len(pred_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)
                matched_gt = set()

                for pred_idx in range(ious.shape[0]):
                    gt_idx = torch.argmax(ious[pred_idx]).item()
                    max_iou = ious[pred_idx, gt_idx].item()

                    if max_iou >= iou_threshold and gt_idx not in matched_gt:
                        matched_gt.add(gt_idx)
                        true_positives += 1
                    else:
                        false_positives += 1

                false_negatives += len(gt_boxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def latest_checkpoint(checkpoint_dir: Path):
    checkpoints = sorted(checkpoint_dir.glob("fasterrcnn_epoch_*.pth"))
    return checkpoints[-1] if checkpoints else None


def load_checkpoint_if_available(model, optimizer, lr_scheduler, checkpoint_dir: Path, explicit_path: str | None, device):
    checkpoint_path = Path(explicit_path) if explicit_path else latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or not checkpoint_path.exists():
        return 0, 0.0, cpu_state_dict(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state_dict = checkpoint.get("lr_scheduler_state_dict")
    if scheduler_state_dict is not None:
        lr_scheduler.load_state_dict(scheduler_state_dict)
    start_epoch = int(checkpoint["epoch"])
    best_map50 = float(checkpoint.get("best_map50", 0.0))
    best_model_wts = cpu_state_dict(model)
    print(f"Resuming from checkpoint: {checkpoint_path} (starting at epoch {start_epoch + 1})")
    return start_epoch, best_map50, best_model_wts


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on the Roboflow tulip dataset.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_dir = ensure_dataset(PROJECT_DIR)
    train_dataset = TulipCocoDetection(
        root=str(dataset_dir / "train"),
        annFile=str(dataset_dir / "train" / "_annotations.coco.json"),
        augment=False,
    )
    valid_dataset = TulipCocoDetection(
        root=str(dataset_dir / "valid"),
        annFile=str(dataset_dir / "valid" / "_annotations.coco.json"),
    )
    test_dataset = TulipCocoDetection(
        root=str(dataset_dir / "test"),
        annFile=str(dataset_dir / "test" / "_annotations.coco.json"),
    )
    print(f"Train: {len(train_dataset)}")
    print(f"Valid: {len(valid_dataset)}")
    print(f"Test: {len(test_dataset)}")

    pin_memory = torch.cuda.is_available()
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
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
    print(f"Dataloaders ready | num_workers={num_workers} | pin_memory={pin_memory}")

    model = build_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    checkpoint_dir = PROJECT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    start_epoch, best_map50, best_model_wts = load_checkpoint_if_available(
        model,
        optimizer,
        lr_scheduler,
        checkpoint_dir,
        args.resume,
        device,
    )

    epochs_without_improvement = 0
    if start_epoch > 0:
        print("Resetting early-stopping counter after resume to give the run room to continue.")

    train_losses = []
    val_map50_scores = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device, non_blocking=True) for img in images]

            fixed_targets = []
            for target in targets:
                fixed_target = {}
                for key, value in target.items():
                    if key == "boxes" and value.dim() == 1 and value.numel() == 0:
                        fixed_target[key] = torch.empty((0, 4), dtype=value.dtype, device=device)
                    else:
                        fixed_target[key] = value.to(device, non_blocking=True)
                fixed_targets.append(fixed_target)

            loss_dict = model(images, fixed_targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{total_batches} | "
                    f"Batch Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        stabilize_cuda()
        checkpoint_path = checkpoint_dir / f"fasterrcnn_epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": cpu_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "best_map50": best_map50,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

        should_validate = ((epoch + 1) % args.validation_interval == 0) or ((epoch + 1) == args.epochs)
        if should_validate:
            stabilize_cuda()
            map_results = evaluate_map50(model, valid_loader, device)
            val_map50 = map_results["map_50"].item()
            val_map50_scores.append(val_map50)

            print(
                f"Epoch {epoch + 1}/{args.epochs} COMPLETE | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val mAP@0.50: {val_map50:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if val_map50 > best_map50:
                best_map50 = val_map50
                best_model_wts = cpu_state_dict(model)
                epochs_without_improvement = 0
                print("  New best model saved.")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} validation check(s).")

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} COMPLETE | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        stabilize_cuda()

    model.load_state_dict(best_model_wts)

    model_output_path = PROJECT_DIR / "fasterrcnn_tulip_improved.pth"
    torch.save(model.state_dict(), model_output_path)
    print(f"Model saved to {model_output_path}")

    map_results = evaluate_map50(model, test_loader, device)
    prf1_results = compute_prf1(model, test_loader, device, score_threshold=0.50, iou_threshold=0.5)

    print("=== Detection Metrics ===")
    print(f"mAP@[0.5:0.95]: {map_results['map'].item():.4f}")
    print(f"mAP@0.50:       {map_results['map_50'].item():.4f}")
    print(f"Precision:      {prf1_results['precision']:.4f}")
    print(f"Recall:         {prf1_results['recall']:.4f}")
    print(f"F1-score:       {prf1_results['f1']:.4f}")
    print(f"TP: {prf1_results['TP']}, FP: {prf1_results['FP']}, FN: {prf1_results['FN']}")


if __name__ == "__main__":
    main()
