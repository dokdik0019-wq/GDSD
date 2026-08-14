# GDSD — Gradient Direction Symmetry Difference Edge Detector

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

## Installation

```bash
git clone <your-repo-url> GDSD
cd GDSD
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.9+ with `numpy` and `opencv-python`.

## Usage

### Command line demo

```bash
# Run on a generated synthetic sample (rectangle, line, circle, text + noise)
python demo.py

# Run on your own image
python demo.py --input photo.jpg

# Tune parameters
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

## Tests

```bash
pytest tests/ -q
```

The test suite covers: edge detection on synthetic shapes, empty
output on flat images, threshold validation, output shapes, and
reproducibility.

## Project structure

```
GDSD/
|-- gdsd.py            core implementation (fit, response, detection)
|-- demo.py            command-line demo + result visualization
|-- tests/
|   `-- test_gdsd.py   pytest suite
|-- requirements.txt
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

## License

Academic use. See the thesis for the full method description and
derivation.
