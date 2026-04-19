import argparse
import sys
from pathlib import Path

import torch


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.dataset import build_dataloaders  # noqa: E402
from common.metrics import compute_prf1, evaluate_map  # noqa: E402
from models.faster_rcnn.utils import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, _, _, test_loader = build_dataloaders(args.dataset_root, batch_size=2)
    model = build_model(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    map_results = evaluate_map(model, test_loader, device)
    prf1_results = compute_prf1(model, test_loader, device)

    print(f"mAP@[0.5:0.95]: {map_results['map'].item():.4f}")
    print(f"mAP@0.50: {map_results['map_50'].item():.4f}")
    print(f"Precision: {prf1_results['precision']:.4f}")
    print(f"Recall: {prf1_results['recall']:.4f}")
    print(f"F1-score: {prf1_results['f1']:.4f}")


if __name__ == "__main__":
    main()
