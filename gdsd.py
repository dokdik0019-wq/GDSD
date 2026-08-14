"""
GDSD -- Gradient Direction Symmetry Difference edge detector
============================================================

Second-order edge detection based on local quadratic surface fitting.
For every pixel the intensity surface is approximated by a quadratic
polynomial fitted with least squares on a 5x5 neighborhood:

    z(x, y) = a x^2 + b y^2 + c x y + d x + e y + f

The GDSD response is the second directional derivative of the fitted
surface taken along the gradient direction (scaled by the squared
gradient magnitude):

    R = 2 a d^2 + 2 c d e + 2 b e^2

where (d, e) are the linear coefficients of the fit, i.e. the gradient
components. Edges are marked where R changes sign (zero crossing)
between the two pixels on either side of the current pixel along the
gradient direction, with an additional percentile contrast constraint
(one neighbor above Tp while the other is below Tlow).

Reference: doctoral thesis, Chapter 3, Section 3.2 (GDSD edge detector).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import cv2
import numpy as np

__all__ = [
    "detect_edges",
    "fit_quadratic_maps",
    "make_demo_image",
    "to_gray_float",
]

ArrayDict = Dict[str, np.ndarray]


def to_gray_float(image: np.ndarray) -> np.ndarray:
    """Convert an image to a float64 grayscale array in the range [0, 1]."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray.astype(np.float64)
    if gray.size and gray.max() > 1.0:
        gray /= 255.0
    return gray


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with a 5x5 kernel (no-op when sigma <= 0)."""
    if sigma <= 0.0:
        return image
    return cv2.GaussianBlur(
        image, (5, 5), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )


@lru_cache(maxsize=None)
def _design_pinv(radius: int, step: float) -> np.ndarray:
    """Precomputed pseudo-inverse of the quadratic design matrix (2*radius+1)^2 x 6."""
    rows = []
    for r in range(-radius, radius + 1):
        for c in range(-radius, radius + 1):
            x = c * step
            y = r * step
            rows.append([x * x, y * y, x * y, x, y, 1.0])
    return np.linalg.pinv(np.asarray(rows, dtype=np.float64))


def fit_quadratic_maps(
    image: np.ndarray,
    radius: int = 2,
    step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a quadratic surface on a (2*radius+1) window around every pixel.

    Returns the six coefficient maps (a, b, c, d, e, f) of
    z = a x^2 + b y^2 + c x y + d x + e y + f.

    The pseudo-inverse is precomputed once (the design matrix only
    depends on the window geometry), so each pixel costs one matrix
    multiply of size 6 x (2*radius+1)^2.
    """
    gray = to_gray_float(image)
    h, w = gray.shape
    pinv = _design_pinv(radius, step)
    padded = np.pad(gray, radius, mode="edge")
    win = 2 * radius + 1

    a = np.zeros((h, w), dtype=np.float64)
    b = np.zeros((h, w), dtype=np.float64)
    c = np.zeros((h, w), dtype=np.float64)
    d = np.zeros((h, w), dtype=np.float64)
    e = np.zeros((h, w), dtype=np.float64)
    f = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            patch = padded[y : y + win, x : x + win].reshape(-1)
            coeffs = pinv @ patch
            a[y, x], b[y, x], c[y, x], d[y, x], e[y, x], f[y, x] = coeffs

    return a, b, c, d, e, f


def quantized_gradient_direction(dx: float, dy: float) -> Tuple[int, int]:
    """Round the gradient direction to the nearest 8-neighbor step (step_y, step_x)."""
    mag = float(np.hypot(dx, dy))
    if mag <= 1e-12:
        return 0, 0
    step_y = int(np.round(dy / mag))
    step_x = int(np.round(dx / mag))
    return step_y, step_x


def detect_edges(
    image: np.ndarray,
    percentile: float = 90.0,
    sigma: float = 1.4,
    gradient_threshold: float = 1e-2,
    radius: int = 2,
    step: float = 0.1,
) -> ArrayDict:
    """Run the GDSD edge detector.

    Args:
        image: Input image (BGR or grayscale, uint8 or float).
        percentile: High threshold percentile of the response (Tp).
            The low threshold is the symmetric percentile (100 - percentile).
            Typical range 80-95; higher means fewer, stronger edges.
        sigma: Gaussian blur sigma applied before the quadratic fit.
        gradient_threshold: Minimum gradient magnitude for a pixel to be
            considered (suppresses flat regions).
        radius: Quadratic fit window radius (1 or 2; 2 = 5x5 window).
        step: Grid spacing used in the design matrix.

    Returns:
        Dict with keys:
            gradient_magnitude, response, valid, neighbor_positive,
            neighbor_negative, percentile_contrast, zero_crossing, edge,
            Tp, Tlow
        where ``edge`` is a boolean map with True on detected edges.
    """
    if not 1.0 <= percentile <= 99.0:
        raise ValueError("percentile must be between 1 and 99")

    gray = to_gray_float(image)
    gray = gaussian_blur(gray, sigma)
    a, b, c, d, e, _ = fit_quadratic_maps(gray, radius=radius, step=step)

    gradient_magnitude = np.hypot(d, e)
    valid = gradient_magnitude > gradient_threshold

    # Second directional derivative along the gradient, scaled by g^2:
    # R = (grad^T H grad) = 2 a d^2 + 2 c d e + 2 b e^2.
    response = 2.0 * a * d * d + 2.0 * c * d * e + 2.0 * b * e * e

    values = response[valid]
    if values.size == 0:
        zeros = np.zeros_like(gray, dtype=bool)
        return {
            "gradient_magnitude": gradient_magnitude,
            "response": response,
            "valid": valid,
            "neighbor_positive": np.zeros_like(response),
            "neighbor_negative": np.zeros_like(response),
            "percentile_contrast": zeros,
            "zero_crossing": zeros,
            "edge": zeros,
            "Tp": np.array(0.0, dtype=np.float64),
            "Tlow": np.array(0.0, dtype=np.float64),
        }

    tp = float(np.percentile(values, percentile))
    tlow = float(np.percentile(values, 100.0 - percentile))

    h, w = gray.shape
    nb1 = np.zeros_like(response)
    nb2 = np.zeros_like(response)
    for y in range(h):
        for x in range(w):
            if not valid[y, x]:
                continue
            step_y, step_x = quantized_gradient_direction(d[y, x], e[y, x])
            if step_y == 0 and step_x == 0:
                continue
            y1 = int(np.clip(y + step_y, 0, h - 1))
            x1 = int(np.clip(x + step_x, 0, w - 1))
            y2 = int(np.clip(y - step_y, 0, h - 1))
            x2 = int(np.clip(x - step_x, 0, w - 1))
            nb1[y, x] = response[y1, x1]
            nb2[y, x] = response[y2, x2]

    percentile_contrast = ((nb1 > tp) & (nb2 < tlow)) | ((nb2 > tp) & (nb1 < tlow))
    zero_crossing = nb1 * nb2 < 0.0
    edge = valid & percentile_contrast & zero_crossing

    return {
        "gradient_magnitude": gradient_magnitude,
        "response": response,
        "valid": valid,
        "neighbor_positive": nb1,
        "neighbor_negative": nb2,
        "percentile_contrast": percentile_contrast,
        "zero_crossing": zero_crossing,
        "edge": edge,
        "Tp": np.array(tp, dtype=np.float64),
        "Tlow": np.array(tlow, dtype=np.float64),
    }


def make_demo_image(size: int = 256, seed: int = 0) -> np.ndarray:
    """Create a synthetic test image: rectangle, diagonal line, circle and text.

    The image contains straight edges at several orientations plus mild
    Gaussian noise, which exercises the zero-crossing logic of GDSD.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (110, 110), 160, -1)
    cv2.line(img, (130, 35), (220, 215), 230, 4)
    cv2.circle(img, (190, 70), 30, 255, -1)
    cv2.putText(img, "GDSD", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 190, 4, cv2.LINE_AA)

    rng = np.random.default_rng(seed)
    noisy = np.clip(img.astype(np.float64) + rng.normal(0.0, 3.0, img.shape), 0, 255)
    return noisy.astype(np.uint8)
