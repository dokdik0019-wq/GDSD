from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from quadratic_fit import quadratic_coeff_maps


def elbow_threshold(values: np.ndarray) -> tuple[float, int]:
    """
    Find the elbow point of a sorted response curve using the maximum
    perpendicular distance to the line between the first and last points.
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 0

    data = np.sort(data)
    if data.size == 1:
        return float(data[0]), 0

    xs = np.linspace(0.0, 100.0, data.size)
    ys = data

    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0.0:
        idx = data.size // 2
        return float(data[idx]), idx

    distances = np.abs((y2 - y1) * xs - (x2 - x1) * ys + x2 * y1 - y2 * x1) / denom
    idx = int(np.argmax(distances))
    return float(data[idx]), idx


def _quantize_theta(theta: np.ndarray) -> np.ndarray:
    angle = (np.degrees(theta) + 180.0) % 180.0
    direction = np.zeros_like(angle, dtype=np.int16)
    direction[(22.5 <= angle) & (angle < 67.5)] = 45
    direction[(67.5 <= angle) & (angle < 112.5)] = 90
    direction[(112.5 <= angle) & (angle < 157.5)] = 135
    return direction


def _shift_with_edge_padding(array: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    padded = np.pad(array, 1, mode="edge")
    return padded[1 + row_shift : 1 + row_shift + array.shape[0], 1 + col_shift : 1 + col_shift + array.shape[1]]


def nonmax_suppression(R: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Keep only response values that are local maxima along the gradient."""
    response = np.asarray(R, dtype=np.float64)
    direction = _quantize_theta(theta)

    left = _shift_with_edge_padding(response, 0, -1)
    right = _shift_with_edge_padding(response, 0, 1)
    up = _shift_with_edge_padding(response, -1, 0)
    down = _shift_with_edge_padding(response, 1, 0)
    up_right = _shift_with_edge_padding(response, -1, 1)
    down_left = _shift_with_edge_padding(response, 1, -1)
    up_left = _shift_with_edge_padding(response, -1, -1)
    down_right = _shift_with_edge_padding(response, 1, 1)

    kept = np.zeros_like(response)

    mask = direction == 0
    kept[mask & (response >= left) & (response >= right)] = response[mask & (response >= left) & (response >= right)]

    mask = direction == 45
    keep_mask = mask & (response >= up_right) & (response >= down_left)
    kept[keep_mask] = response[keep_mask]

    mask = direction == 90
    keep_mask = mask & (response >= up) & (response >= down)
    kept[keep_mask] = response[keep_mask]

    mask = direction == 135
    keep_mask = mask & (response >= up_left) & (response >= down_right)
    kept[keep_mask] = response[keep_mask]

    return kept


def hysteresis_threshold(nms: np.ndarray, high: float, low: float) -> np.ndarray:
    """Keep strong pixels and weak pixels connected to strong pixels."""
    data = np.asarray(nms, dtype=np.float64)
    strong = data >= high
    weak = (data >= low) & ~strong

    if not np.any(strong):
        return np.zeros_like(data, dtype=np.uint8)

    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    candidate = strong | weak
    labels, count = ndi.label(candidate, structure=structure)
    strong_labels = np.unique(labels[strong])
    keep = np.isin(labels, strong_labels)
    keep[labels == 0] = False
    return keep.astype(np.uint8) * 255


def else_single_threshold(img: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = quadratic_coeff_maps(img, radius=1, step=0.1)
    R = np.sqrt(D * D + E * E)
    threshold, _ = elbow_threshold(R.ravel())
    edge = (R >= threshold).astype(np.uint8) * 255
    return edge


def else_double_threshold(img: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = quadratic_coeff_maps(img, radius=1, step=0.1)
    R = np.sqrt(D * D + E * E)
    theta = np.arctan2(E, D)

    nms = nonmax_suppression(R, theta)
    values = nms[nms > 0]
    if values.size == 0:
        return np.zeros_like(R, dtype=np.uint8)

    high, high_idx = elbow_threshold(values)
    percentile_grid = np.linspace(0.0, 100.0, values.size)
    high_percentile = float(percentile_grid[min(high_idx, percentile_grid.size - 1)])
    low_percentile = max(high_percentile - 4.0, 0.0)
    low = float(np.percentile(values, low_percentile))
    return hysteresis_threshold(nms, high, low)
