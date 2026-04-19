import argparse
import sys
from pathlib import Path

import torch
import torchvision
from PIL import Image


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.retinanet.utils import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run RetinaNet prediction on a single image.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = torchvision.transforms.functional.to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    keep = output["scores"].detach().cpu() >= args.threshold
    print("Boxes:", output["boxes"].detach().cpu()[keep])
    print("Scores:", output["scores"].detach().cpu()[keep])


if __name__ == "__main__":
    main()
