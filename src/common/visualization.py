from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def show_image(image_path: str | Path, title: str | None = None):
    image = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()
