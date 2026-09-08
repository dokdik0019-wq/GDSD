from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image as a float32 grayscale array in [0, 1]."""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    return array / 255.0


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert an array to uint8 for saving or display."""
    array = np.asarray(img)
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        if not np.any(finite):
            return np.zeros(array.shape, dtype=np.uint8)
        data = array.copy()
        min_value = float(np.nanmin(data[finite]))
        max_value = float(np.nanmax(data[finite]))
        if max_value > min_value:
            data = (data - min_value) / (max_value - min_value)
        else:
            data = np.zeros_like(data)
        return np.clip(data * 255.0, 0, 255).astype(np.uint8)

    data = array.astype(np.float32)
    if np.max(data) <= 1.0:
        data = data * 255.0
    return np.clip(data, 0, 255).astype(np.uint8)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save a grayscale or RGB image using Pillow."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(img)
    if array.ndim == 2:
        Image.fromarray(normalize_to_uint8(array), mode="L").save(output_path)
        return

    if array.ndim == 3 and array.shape[2] == 3:
        Image.fromarray(normalize_to_uint8(array), mode="RGB").save(output_path)
        return

    raise ValueError("save_image expects a 2-D grayscale array or a 3-D RGB array")


def show_images(images: list[np.ndarray], titles: list[str]) -> None:
    """Show a small grid of images with Matplotlib."""
    import matplotlib.pyplot as plt

    count = len(images)
    if count == 0:
        return

    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))
    if count == 1:
        axes = [axes]

    for axis, image, title in zip(axes, images, titles):
        array = np.asarray(image)
        if array.ndim == 2:
            axis.imshow(array, cmap="gray")
        else:
            axis.imshow(normalize_to_uint8(array))
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    plt.show()
