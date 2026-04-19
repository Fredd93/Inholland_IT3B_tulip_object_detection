import sys
from pathlib import Path

import torch
import torchvision


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from common.metrics import cpu_state_dict  # noqa: E402


def build_model(device: torch.device):
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    except Exception as exc:
        print(f"Falling back to randomly initialized weights: {exc}")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        2,
    )
    model.to(device)
    return model


def latest_checkpoint(checkpoint_dir: Path):
    checkpoints = sorted(checkpoint_dir.glob("faster_rcnn_epoch_*.pth"))
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
