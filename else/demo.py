from __future__ import annotations

import argparse
from pathlib import Path

from else_detector import else_double_threshold, else_single_threshold
from utils import load_grayscale, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ELSE edge detector demo (IJ MCS 2026)")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument(
        "--method",
        required=True,
        choices=["else_single", "else_double"],
        help="Single-threshold (elbow) or double-threshold (elbow + NMS + hysteresis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = load_grayscale(image_path)
    stem = image_path.stem

    if args.method == "else_single":
        result = else_single_threshold(gray)
        suffix = "else_single"
    else:
        result = else_double_threshold(gray)
        suffix = "else_double"

    output_path = output_dir / f"{stem}_{suffix}.png"
    save_image(output_path, result)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
