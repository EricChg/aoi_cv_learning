from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import ensure_dir, list_images, load_config, read_image, write_yaml


def object_points(board_cols: int, board_rows: int, square_size_mm: float) -> np.ndarray:
    points = np.zeros((board_rows * board_cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    return points * square_size_mm


def find_corners(image: np.ndarray, board_size: tuple[int, int]) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def reprojection_error(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> float:
    total_error = 0.0
    total_points = 0
    for obj_points, img_points, rvec, tvec in zip(object_points_list, image_points_list, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, distortion_coefficients)
        error = cv2.norm(img_points, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(obj_points)
    return float(np.sqrt(total_error / max(total_points, 1)))


def approximate_pixel_per_mm(
    image_points_list: list[np.ndarray],
    board_cols: int,
    square_size_mm: float,
) -> float | None:
    distances = []
    for corners in image_points_list:
        corner_grid = corners.reshape(-1, 2)
        for row_start in range(0, len(corner_grid), board_cols):
            row = corner_grid[row_start : row_start + board_cols]
            for left, right in zip(row[:-1], row[1:]):
                distances.append(float(np.linalg.norm(right - left)))
    if not distances:
        return None
    return float(np.median(distances) / square_size_mm)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return np.hstack([left, right])


def run_calibration(config: dict[str, Any], input_path: str, output_path: str) -> None:
    calibration = config.get("calibration", {}) or {}
    board_cols = int(calibration.get("board_cols") or 9)
    board_rows = int(calibration.get("board_rows") or 6)
    square_size_mm = float(calibration.get("square_size_mm") or 25.0)
    min_valid_images = int(calibration.get("min_valid_images") or 3)
    board_size = (board_cols, board_rows)
    object_template = object_points(board_cols, board_rows, square_size_mm)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    valid_images: list[Path] = []
    skipped_images: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in list_images(input_path):
        image = read_image(path)
        if image is None:
            skipped_images.append(str(path))
            continue
        image_size = (image.shape[1], image.shape[0])
        corners = find_corners(image, board_size)
        if corners is None:
            skipped_images.append(str(path))
            continue
        obj_points.append(object_template.copy())
        img_points.append(corners)
        valid_images.append(path)

    if len(valid_images) < min_valid_images or image_size is None:
        raise RuntimeError(
            f"Need at least {min_valid_images} valid chessboard images, found {len(valid_images)}. "
            "Capture more images with clearly visible corners."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )
    error = reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    pixel_per_mm = approximate_pixel_per_mm(img_points, board_cols, square_size_mm)

    output_dir = ensure_dir(output_path)
    undistorted_dir = ensure_dir(output_dir / "undistorted")
    comparison_dir = ensure_dir(output_dir / "comparison")
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, image_size, 1, image_size)

    for path in valid_images:
        image = read_image(path)
        if image is None:
            continue
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        cv2.imwrite(str(undistorted_dir / f"{path.stem}_undistorted.png"), undistorted)
        cv2.imwrite(str(comparison_dir / f"{path.stem}_comparison.png"), side_by_side(image, undistorted))

    write_yaml(
        output_dir / "calibration_result.yaml",
        {
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "board_size": {"cols": board_cols, "rows": board_rows},
            "square_size_mm": square_size_mm,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.ravel().tolist(),
            "rms": float(rms),
            "reprojection_error": error,
            "valid_images": [str(path) for path in valid_images],
            "skipped_images": skipped_images,
            "approximate_pixel_per_mm": pixel_per_mm,
        },
    )
    print(f"Wrote calibration outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV camera calibration practice.")
    parser.add_argument("--input", required=True, help="Directory containing chessboard images.")
    parser.add_argument("--config", default="examples/traditional_aoi/sample_config.yaml")
    parser.add_argument("--output", default="outputs/traditional_aoi/calibration")
    args = parser.parse_args()
    run_calibration(load_config(args.config), args.input, args.output)


if __name__ == "__main__":
    main()
