"""Tests for the GDSD edge detector."""

import numpy as np
import pytest

from gdsd import detect_edges, fit_quadratic_maps, make_demo_image, soft_edge_map


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


def test_sigma_kernel_not_saturated():
    # Regression (2026-09-08 review): a fixed 5x5 kernel caps effective
    # sigma ~1.38, so sigma=1.4 and sigma=2.8 produced identical outputs.
    # Kernel size must be derived from sigma (ksize=(0,0)); with that fix
    # the two sigmas must genuinely differ on a noisy edge image.
    img = make_demo_image(seed=3)
    e14 = detect_edges(img, sigma=1.4)["edge"]
    e28 = detect_edges(img, sigma=2.8)["edge"]
    assert not np.array_equal(e14, e28)
    # sanity: both still detect a reasonable number of edges
    assert int(e14.sum()) > 50
    assert int(e28.sum()) > 50


def test_thinning_zcnms_reduces_zc_without_touching_strength():
    img = make_demo_image(seed=7)
    s_plain = soft_edge_map(img, strength="gradient_magnitude", thinning="none")
    s_zcnms = soft_edge_map(img, strength="gradient_magnitude", thinning="zcnms")
    assert (s_zcnms > 0).sum() <= (s_plain > 0).sum()
    # strength of surviving pixels unchanged (same gm values)
    common = (s_plain > 0) & (s_zcnms > 0)
    assert np.allclose(s_plain[common], s_zcnms[common])
    # at least some pixels removed on the demo image (doublet flanks)
    assert (s_zcnms > 0).sum() < (s_plain > 0).sum()


def test_soft_edge_map_shapes_and_masks():
    img = make_demo_image()
    s1 = soft_edge_map(img, strength="response")
    s2 = soft_edge_map(img, strength="gradient_magnitude")
    assert s1.shape == img.shape and s1.dtype == np.float32
    assert s2.shape == img.shape and s2.dtype == np.float32
    # same zero-crossing support; differ only in strength
    assert np.array_equal(s1 > 0, s2 > 0)
    assert (s1 > 0).sum() > 0
    # every soft zero crossing carries a nonzero strength
    assert (s1[s1 > 0] > 0).all()


def test_soft_edge_map_strength_choice():
    img = make_demo_image()
    s1 = soft_edge_map(img, strength="response")
    s2 = soft_edge_map(img, strength="gradient_magnitude")
    # strengths genuinely differ (not accidentally identical)
    nz = s1 > 0
    assert not np.allclose(s1[nz], s2[nz])


def test_soft_edge_map_invalid_strength():
    img = make_demo_image()
    with pytest.raises(ValueError):
        soft_edge_map(img, strength="bogus")


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
