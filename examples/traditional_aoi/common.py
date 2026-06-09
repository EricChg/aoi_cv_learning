from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def list_images(path: str | Path) -> list[Path]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source.suffix}")
        return [source]
    images = [
        item
        for item in sorted(source.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No readable image files found in: {source}")
    return images


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def roi_from_config(config: dict[str, Any], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    roi = config.get("roi", {}) or {}
    x = int(roi.get("x") or 0)
    y = int(roi.get("y") or 0)
    roi_width = roi.get("width")
    roi_height = roi.get("height")
    w = width - x if roi_width is None else int(roi_width)
    h = height - y if roi_height is None else int(roi_height)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError(
            "Invalid ROI "
            f"(x={x}, y={y}, width={w}, height={h}) for image size "
            f"(width={width}, height={height})"
        )
    return x, y, w, h


def crop_roi(image: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, int]]:
    x, y, width, height = roi_from_config(config, image.shape)
    return image[y : y + height, x : x + width], {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def odd_kernel(value: int | None, minimum: int = 1) -> int:
    kernel = max(minimum, int(value or minimum))
    return kernel if kernel % 2 == 1 else kernel + 1


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_yaml(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def overlay_heatmap(base_gray: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_u8 = normalize_to_uint8(heatmap)
    color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(normalize_to_uint8(base_gray), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.65, color, 0.35, 0)
