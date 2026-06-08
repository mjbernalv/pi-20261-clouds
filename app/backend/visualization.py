from __future__ import annotations
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io

LABEL_COLORS = {
    "Fish":   "red",
    "Flower": "green",
    "Gravel": "blue",
    "Sugar":  "#FFD700",
}


def _rle_to_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    if not rle_string or rle_string.strip() == "":
        return np.zeros((height, width), dtype=np.float32)
    nums = [int(x) for x in rle_string.split()]
    pairs = np.array(nums).reshape(-1, 2)
    flat = np.zeros(height * width, dtype=np.uint8)
    for index, length in pairs:
        index -= 1
        flat[index : index + length] = 255
    return flat.reshape(width, height).T / 255.0


def plot_segmentation(image_bytes: bytes, result: dict) -> plt.Figure:
    """Return a matplotlib Figure with all masks overlaid on a single image."""
    h, w = result["height"], result["width"]
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((w, h), Image.BILINEAR)
    img_np = np.asarray(img, dtype=np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_np)

    legend_patches = []
    for label, color in LABEL_COLORS.items():
        rle = result["predictions"].get(label, {}).get("rle", "")
        mask = _rle_to_mask(rle, h, w)
        if mask.any():
            r, g, b = mcolors.to_rgb(color)
            rgba = np.zeros((h, w, 4), dtype=np.float32)
            rgba[mask > 0] = [r, g, b, 0.45]
            ax.imshow(rgba)
        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.8)
    ax.axis("off")
    plt.tight_layout()
    return fig
