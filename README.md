# GDSD — Gradient-Direction Second Derivative Edge Detector

[![CI](https://github.com/dokdik0019-wq/GDSD/actions/workflows/ci.yml/badge.svg)](https://github.com/dokdik0019-wq/GDSD/actions/workflows/ci.yml)

A second-order edge detection method based on local quadratic surface
fitting. For every pixel, the intensity surface is approximated by a
quadratic polynomial fitted with least squares on a 5x5 neighborhood.
The edge response is the second directional derivative of the fitted
surface along the gradient direction; edges are marked where this
response changes sign (zero crossing) between the two neighboring
pixels along the gradient, with an additional percentile contrast
constraint.

This implementation follows the GDSD method as described in the
author's doctoral thesis, Chapter 3, Section 3.2.

## Quick start

```bash
git clone <your-repo-url> GDSD
cd GDSD
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python demo.py
pytest tests/ -q
```

`demo.py` generates a synthetic test image (rectangle, diagonal line,
circle, text, plus mild noise), runs the detector, and writes the
results to `output/`:

```
[OK] GDSD finished in 0.51s on 256x256
     edge pixels : 1373
     Tp          : 7.204427
     Tlow        : -6.991019
```

Expected result previews are committed under `examples/`:

| File                      | Contents                                |
|---------------------------|-----------------------------------------|
| `examples/sample_input.png`    | generated input image             |
| `examples/sample_edge.png`     | binary edge map (white on black)  |
| `examples/sample_overlay.png`  | input with detected edges in red  |
| `examples/sample_response.png` | normalized GDSD response          |

## Requirements

- Python 3.9+ (tested on 3.10-3.12)
- `numpy` and `opencv-python-headless` (installed automatically by pip)

The project uses `opencv-python-headless`, which provides the full image
processing API without GUI dependencies, so it works on servers, CI, and
headless machines. If you need `cv2.imshow`, install `opencv-python`
instead.

## Installation options

Editable install (recommended for development):

```bash
pip install -e ".[dev]"
```

Regular install:

```bash
pip install .
```

Either install also provides a `gdsd-demo` command:

```bash
gdsd-demo                          # run on the generated sample
gdsd-demo --input photo.jpg        # run on your own image
gdsd-demo --percentile 95 --sigma 1.0
```

## Usage

### Command line

```bash
python demo.py                            # generated sample
python demo.py --input photo.jpg          # your image
python demo.py --percentile 95 --sigma 1.0 --gradient-threshold 0.01
```

Outputs are written to `output/`:

| File                     | Contents                                   |
|--------------------------|--------------------------------------------|
| `<name>_edge.png`        | binary edge map (white on black)           |
| `<name>_overlay.png`     | input image with detected edges in red     |
| `<name>_response.png`    | normalized GDSD response                   |

### Python API

```python
import cv2
from gdsd import detect_edges

image = cv2.imread("photo.jpg")
result = detect_edges(image)          # defaults: percentile=90, sigma=1.4
edges = result["edge"]                # boolean map, True on edges
print(result["Tp"], result["Tlow"])   # the two thresholds used
```

`detect_edges` returns a dict with keys:
`gradient_magnitude`, `response`, `valid`, `neighbor_positive`,
`neighbor_negative`, `percentile_contrast`, `zero_crossing`, `edge`,
`Tp`, `Tlow`.

## How it works

1. **Preprocessing** — convert to grayscale and apply a Gaussian blur
   (`sigma = 1.4` by default) to suppress noise.
2. **Local quadratic fit** — fit

   ```
   z(x, y) = a x^2 + b y^2 + c x y + d x + e y + f
   ```

   by least squares on a 5x5 window around every pixel. The
   pseudo-inverse of the design matrix is precomputed once, so each
   pixel costs a single matrix multiply.

3. **Response** — the GDSD response is the second directional
   derivative of the fit along the gradient direction (scaled by the
   squared gradient magnitude):

   ```
   R = 2 a d^2 + 2 c d e + 2 b e^2
   ```

   where `(d, e)` are the linear coefficients, i.e. the gradient
   components. `R` is positive on one side of an edge and negative on
   the other, which is what the zero-crossing step exploits.

4. **Thresholds** — the high threshold `Tp` is the response value at
   the `percentile` percentile of valid pixels (default 90), and the
   low threshold `Tlow` is the symmetric percentile (default 10).

5. **Zero crossing + contrast** — for each pixel, the two neighbors
   along the quantized gradient direction are compared: the pixel is
   an edge if the response changes sign between the neighbors and one
   neighbor is above `Tp` while the other is below `Tlow`.

## Tests

```bash
pytest tests/ -q
```

The suite covers edge detection on synthetic shapes, empty output on
flat images, threshold validation, output shapes, and reproducibility.

`tests/verify_thesis_table44.py` reproduces the benchmark protocol of
the thesis (radial square pattern, one-pixel tolerance, precision /
recall / F-measure). It needs the thesis test assets on your machine;
without them it prints instructions and exits.

## Project structure

```
GDSD/
|-- gdsd.py            core implementation (fit, response, detection)
|-- demo.py            command-line demo + result visualization
|-- tests/
|   |-- test_gdsd.py               pytest suite
|   `-- verify_thesis_table44.py   thesis benchmark verification
|-- examples/          expected demo outputs (committed)
|-- .github/workflows/ci.yml  CI: tests on 3 OS x Python 3.10-3.12
|-- pyproject.toml     package metadata + `gdsd-demo` entry point
|-- requirements.txt
|-- Makefile           setup / demo / test / clean
|-- LICENSE            MIT
|-- README.md
`-- .gitignore
```

## Parameters

| Parameter            | Default | Meaning                                          |
|----------------------|---------|--------------------------------------------------|
| `percentile`         | 90.0    | high threshold percentile of the response (Tp)   |
| `sigma`              | 1.4     | Gaussian blur sigma before the fit               |
| `gradient_threshold` | 0.01    | minimum gradient magnitude to consider a pixel   |
| `radius`             | 2       | fit window radius (2 = 5x5)                      |

## Continuous integration

Every push runs the test suite and the demo smoke test on
Ubuntu, macOS, and Windows with Python 3.10, 3.11, and 3.12
(`.github/workflows/ci.yml`).

## Citation

If you use this implementation in your work, please cite the paper that
introduces the GDSD method:

> Sinlapakorn T, Puphasuk P, Wetweerapong J. Edge detection using the
> gradient-direction second derivative of a local 2D quadratic
> approximation. Proceedings of the 30th Annual Meeting in Mathematics
> and Conference in Number Theory and Applications 2026. 2026.
> pp. 449–458. ISBN 978-616-438-985-4.

BibTeX:

```bibtex
@inproceedings{sinlapakorn2026gdsd,
  author    = {Sinlapakorn, Thapakorn and Puphasuk, Pikul and Wetweerapong, Jeerayut},
  title     = {Edge detection using the gradient-direction second derivative
               of a local 2D quadratic approximation},
  booktitle = {Proceedings of the 30th Annual Meeting in Mathematics and
               Conference in Number Theory and Applications 2026},
  year      = {2026},
  pages     = {449--458},
  isbn      = {978-616-438-985-4}
}
```

The method is also described in the author's doctoral thesis,
Chapter 3, Section 3.2.

## License

MIT. See `LICENSE`.
