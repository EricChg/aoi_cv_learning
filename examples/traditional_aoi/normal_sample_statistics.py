from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    crop_roi,
    ensure_dir,
    list_images,
    load_config,
    normalize_to_uint8,
    overlay_heatmap,
    read_image,
    to_gray,
    write_json,
)


def prepare_sample(path: Path, config: dict[str, Any]) -> np.ndarray | None:
    image = read_image(path)
    if image is None:
        return None
    roi_image, _ = crop_roi(image, config)
    gray = to_gray(roi_image)
    normal_stats = config.get("normal_stats", {}) or {}
    resize_width = normal_stats.get("resize_width")
    resize_height = normal_stats.get("resize_height")
    if resize_width is not None and resize_height is not None:
        gray = cv2.resize(gray, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)


def validate_shapes(samples: list[np.ndarray], paths: list[Path], resize_configured: bool) -> None:
    shapes = {sample.shape for sample in samples}
    if len(shapes) <= 1:
        return
    shape_report = {str(path): list(sample.shape) for path, sample in zip(paths, samples)}
    hint = "Configure normal_stats.resize_width and resize_height to force a common size."
    if resize_configured:
        hint = "Check ROI and resize settings; samples are still inconsistent."
    raise ValueError(f"Normal sample dimensions do not match: {shape_report}. {hint}")


def run_statistics(config: dict[str, Any], input_path: str, output_path: str) -> None:
    image_paths = list_images(input_path)
    samples: list[np.ndarray] = []
    valid_paths: list[Path] = []
    skipped: list[str] = []
    for path in image_paths:
        sample = prepare_sample(path, config)
        if sample is None:
            skipped.append(str(path))
            continue
        samples.append(sample)
        valid_paths.append(path)

    if not samples:
        raise RuntimeError("No readable normal samples found.")

    normal_stats = config.get("normal_stats", {}) or {}
    resize_configured = normal_stats.get("resize_width") is not None and normal_stats.get("resize_height") is not None
    validate_shapes(samples, valid_paths, resize_configured)

    stack = np.stack(samples, axis=0)
    mean_map = stack.mean(axis=0)
    variance_map = stack.var(axis=0)
    std_map = stack.std(axis=0)

    output_dir = ensure_dir(output_path)
    diff_dir = ensure_dir(output_dir / "diff_overlays")
    cv2.imwrite(str(output_dir / "mean.png"), normalize_to_uint8(mean_map))
    cv2.imwrite(str(output_dir / "variance.png"), normalize_to_uint8(variance_map))
    cv2.imwrite(str(output_dir / "std.png"), normalize_to_uint8(std_map))

    diff_values = []
    per_image = []
    for path, sample in zip(valid_paths, samples):
        diff = np.abs(sample - mean_map)
        diff_values.extend(diff.ravel().tolist())
        cv2.imwrite(str(diff_dir / f"{path.stem}_diff.png"), normalize_to_uint8(diff))
        cv2.imwrite(str(diff_dir / f"{path.stem}_diff_overlay.png"), overlay_heatmap(sample, diff))
        per_image.append(
            {
                "source": str(path),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )

    diff_array = np.array(diff_values, dtype=np.float32)
    write_json(
        output_dir / "summary.json",
        {
            "sample_count": len(samples),
            "skipped_images": skipped,
            "image_shape": {"height": int(samples[0].shape[0]), "width": int(samples[0].shape[1])},
            "average_std": float(std_map.mean()),
            "p95_diff": float(np.percentile(diff_array, 95)),
            "p99_diff": float(np.percentile(diff_array, 99)),
            "per_image": per_image,
        },
    )
    print(f"Wrote normal sample statistics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute normal sample mean, variance, and diff maps.")
    parser.add_argument("--input", required=True, help="Directory containing normal sample images.")
    parser.add_argument("--config", default="examples/traditional_aoi/sample_config.yaml")
    parser.add_argument("--output", default="outputs/traditional_aoi/normal_stats")
    args = parser.parse_args()
    run_statistics(load_config(args.config), args.input, args.output)


if __name__ == "__main__":
    main()
