from __future__ import annotations

from functools import lru_cache

import numpy as np


def _coordinate_grid(radius: int, step: float) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) * step
    x_grid, y_grid = np.meshgrid(offsets, offsets)
    return x_grid.ravel(), y_grid.ravel()


@lru_cache(maxsize=16)
def _pseudoinverse(radius: int, step: float) -> np.ndarray:
    """Precompute the pseudoinverse of the quadratic design matrix."""
    x, y = _coordinate_grid(radius, step)
    design = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    return np.linalg.solve(design.T @ design, design.T)


def quadratic_coeff_maps(img: np.ndarray, radius: int = 1, step: float = 0.1):
    """
    Fit a local quadratic surface to each pixel neighbourhood.

    Surface model:
        S(x, y) = A x^2 + B y^2 + C x y + D x + E y + F
    """
    if img.ndim != 2:
        raise ValueError("img must be a 2-D grayscale array")

    image = np.asarray(img, dtype=np.float64)
    pinv = _pseudoinverse(radius, float(step))
    window = 2 * radius + 1

    padded = np.pad(image, radius, mode="edge")
    height, width = image.shape

    coeffs = [np.empty((height, width), dtype=np.float64) for _ in range(6)]

    for row in range(height):
        for col in range(width):
            patch = padded[row : row + window, col : col + window].reshape(-1)
            A, B, C, D, E, F = pinv @ patch
            coeffs[0][row, col] = A
            coeffs[1][row, col] = B
            coeffs[2][row, col] = C
            coeffs[3][row, col] = D
            coeffs[4][row, col] = E
            coeffs[5][row, col] = F

    return tuple(coeffs)

