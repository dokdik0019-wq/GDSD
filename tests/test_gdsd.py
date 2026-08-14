"""Tests for the GDSD edge detector."""

import numpy as np
import pytest

from gdsd import detect_edges, fit_quadratic_maps, make_demo_image


def test_demo_image_has_shapes():
    img = make_demo_image()
    assert img.shape == (256, 256)
    assert img.dtype == np.uint8
    assert img.max() > 200  # bright shapes present
    assert img.std() > 0.0  # not a constant image


def test_flat_image_no_edges():
    flat = np.full((64, 64), 128.0, dtype=np.float64)
    result = detect_edges(flat)
    assert result["edge"].sum() == 0


def test_rectangle_edges_detected():
    # A perfectly binary rectangle degenerates the percentile thresholds
    # (Tp/Tlow pin to the exact min/max, same as the thesis implementation),
    # so use mild noise to keep the response distribution realistic.
    img = np.zeros((128, 128), dtype=np.float64)
    img[32:96, 32:96] = 1.0
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0.0, 0.01, img.shape), 0, 1)
    result = detect_edges(img)
    edge_pixels = int(result["edge"].sum())
    assert edge_pixels > 50


def test_percentile_validation():
    img = make_demo_image()
    with pytest.raises(ValueError):
        detect_edges(img, percentile=0.5)
    with pytest.raises(ValueError):
        detect_edges(img, percentile=100.5)


def test_output_keys_and_shapes():
    img = make_demo_image()
    result = detect_edges(img)
    for key in ("response", "edge", "Tp", "Tlow"):
        assert key in result
    assert result["response"].shape == img.shape
    assert result["edge"].shape == img.shape
    assert result["edge"].dtype == np.bool_
    assert result["Tp"] > 0.0
    assert result["Tlow"] < 0.0


def test_reproducible():
    img = make_demo_image(seed=7)
    r1 = detect_edges(img)
    r2 = detect_edges(img)
    assert np.array_equal(r1["edge"], r2["edge"])


def test_quadratic_fit_on_plane():
    # f(x, y) = 0.05 + 0.01 * (pixel column). The design matrix uses
    # x = 0.1 * column, so the fitted linear coefficient must be
    # d = 0.01 / 0.1 = 0.1 and the constant term f = 0.05 + 0.01 * col.
    # Values stay in [0, 1] so to_gray_float does not rescale them.
    y, x = np.mgrid[0:32, 0:32]
    img = (0.05 + 0.01 * x).astype(np.float64)
    a, b, c, d, e, f = fit_quadratic_maps(img)
    assert abs(d[16, 16] - 0.1) < 1e-6
    assert abs(e[16, 16]) < 1e-6
    assert abs(a[16, 16]) < 1e-6
    assert abs(b[16, 16]) < 1e-6
    assert abs(f[16, 16] - 0.21) < 1e-6


def test_noise_robustness_demo_image():
    # Mild noise must not destroy all detections.
    img = make_demo_image(seed=42)
    result = detect_edges(img)
    assert int(result["edge"].sum()) > 100
