from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import crop_roi, ensure_dir, list_images, load_config, odd_kernel, read_image, to_gray, write_json


def apply_preprocess(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    preprocess = config.get("preprocess", {}) or {}
    gray = to_gray(image)

    blur_kernel = int(preprocess.get("blur_kernel") or 0)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (odd_kernel(blur_kernel), odd_kernel(blur_kernel)), 0)

    illumination = preprocess.get("illumination", {}) or {}
    method = str(illumination.get("method") or "none").lower()
    background_kernel = odd_kernel(illumination.get("background_kernel") or 51, minimum=3)

    if method == "divide":
        background = cv2.GaussianBlur(gray, (background_kernel, background_kernel), 0)
        return cv2.divide(gray, background, scale=255)
    if method == "subtract":
        background = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_kernel, background_kernel)),
        )
        corrected = cv2.subtract(gray, background)
        return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def build_candidate_mask(gray: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    threshold = config.get("threshold", {}) or {}
    method = str(threshold.get("method") or "otsu").lower()
    invert = bool(threshold.get("invert", False))
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if method == "fixed":
        fixed_value = int(threshold.get("fixed_value") or 40)
        _, mask = cv2.threshold(gray, fixed_value, 255, threshold_type)
    elif method == "adaptive":
        block_size = odd_kernel(threshold.get("adaptive_block_size") or 35, minimum=3)
        c_value = int(threshold.get("adaptive_c") or 5)
        adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        mask = cv2.adaptiveThreshold(gray, 255, adaptive_type, threshold_type, block_size, c_value)
    else:
        _, mask = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)

    edges_config = config.get("edges", {}) or {}
    edges = cv2.Canny(
        gray,
        int(edges_config.get("canny_low") or 50),
        int(edges_config.get("canny_high") or 150),
    )
    if edges_config.get("include_edges", False):
        mask = cv2.bitwise_or(mask, edges)

    morphology = config.get("morphology", {}) or {}
    open_kernel = int(morphology.get("open_kernel") or 0)
    close_kernel = int(morphology.get("close_kernel") or 0)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, edges


def connected_component_candidates(
    mask: np.ndarray,
    gray: np.ndarray,
    roi: dict[str, int],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cc_config = config.get("connected_components", {}) or {}
    min_area = float(cc_config.get("min_area") or 0)
    max_area = cc_config.get("max_area")
    min_width = int(cc_config.get("min_width") or 0)
    min_height = int(cc_config.get("min_height") or 0)
    max_aspect_ratio = cc_config.get("max_aspect_ratio")

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area:
            continue
        if max_area is not None and area > float(max_area):
            continue
        if width < min_width or height < min_height:
            continue
        aspect_ratio = max(width / max(height, 1), height / max(width, 1))
        if max_aspect_ratio is not None and aspect_ratio > float(max_aspect_ratio):
            continue

        component_pixels = gray[labels == label]
        centroid_x, centroid_y = centroids[label]
        candidates.append(
            {
                "label": int(label),
                "bbox": {
                    "x": int(x + roi["x"]),
                    "y": int(y + roi["y"]),
                    "width": int(width),
                    "height": int(height),
                },
                "bbox_roi": {"x": int(x), "y": int(y), "width": int(width), "height": int(height)},
                "area": int(area),
                "centroid": {
                    "x": float(centroid_x + roi["x"]),
                    "y": float(centroid_y + roi["y"]),
                },
                "mean_intensity": float(component_pixels.mean()) if component_pixels.size else 0.0,
                "aspect_ratio": float(aspect_ratio),
            }
        )
    return candidates


def draw_overlay(image: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    for candidate in candidates:
        bbox = candidate["bbox"]
        x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            str(candidate["label"]),
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def process_image(path: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    image = read_image(path)
    if image is None:
        return {"source": str(path), "error": "Image could not be read", "candidates": []}

    roi_image, roi = crop_roi(image, config)
    gray = apply_preprocess(roi_image, config)
    mask_roi, edges = build_candidate_mask(gray, config)
    candidates = connected_component_candidates(mask_roi, gray, roi, config)

    masks_dir = ensure_dir(output_dir / "masks")
    overlays_dir = ensure_dir(output_dir / "overlays")
    edges_dir = ensure_dir(output_dir / "edges")

    stem = path.stem
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[roi["y"] : roi["y"] + roi["height"], roi["x"] : roi["x"] + roi["width"]] = mask_roi
    cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), full_mask)
    cv2.imwrite(str(edges_dir / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), draw_overlay(image, candidates))

    return {
        "source": str(path),
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "roi": roi,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional OpenCV AOI baseline.")
    parser.add_argument("--input", required=True, help="Input image or image directory.")
    parser.add_argument("--config", default="examples/traditional_aoi/sample_config.yaml")
    parser.add_argument("--output", default="outputs/traditional_aoi/baseline")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output)
    results = []
    for image_path in list_images(args.input):
        results.append(process_image(image_path, config, output_dir))

    write_json(output_dir / "candidates.json", {"images": results})
    print(f"Wrote baseline outputs to {output_dir}")


if __name__ == "__main__":
    main()
