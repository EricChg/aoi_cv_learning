# Design: Traditional AOI baseline practice

## Overview

本变更提供三个相互独立但概念连续的 OpenCV 练习：

1. `traditional_aoi_baseline`：从单张或目录图片中生成候选缺陷区域。
2. `camera_calibration`：从棋盘格标定图中计算内参和畸变参数，并执行 undistort。
3. `normal_sample_statistics`：从正常样本中生成 mean map、variance map、diff map，理解正常波动。

实现应优先保持脚本可读、参数清晰、输出可复现，避免过早抽象成复杂框架。

## Proposed Structure

建议新增目录：

```text
examples/
  traditional_aoi/
    README.md
    traditional_aoi_baseline.py
    camera_calibration.py
    normal_sample_statistics.py
    sample_config.yaml
```

建议输出目录：

```text
outputs/
  traditional_aoi/
    baseline/
      candidates.json
      overlays/
      masks/
    calibration/
      calibration_result.yaml
      undistorted/
      comparison/
    normal_stats/
      mean.png
      variance.png
      std.png
      diff_overlays/
      summary.json
```

## Traditional AOI Baseline

处理流程：

- 读取输入图片，支持单图和目录。
- 按配置裁剪 ROI，未配置时使用全图。
- 转灰度，并提供可选 blur。
- 光照归一化：使用大核 Gaussian blur 或 morphological opening 估计背景，再做除法归一化或差分校正。
- 候选区域生成：
  - 阈值：支持 Otsu、自适应阈值、固定阈值。
  - 边缘：支持 Canny 作为辅助可视化或候选生成。
  - 形态学：open/close 去噪和连接断裂区域。
  - 连通域：输出 bbox、area、centroid、mean intensity。
- 后处理：按面积、宽高、长宽比过滤候选。
- 输出：二值 mask、overlay 图片、`candidates.json`。

输出坐标应明确区分 ROI 内坐标和原图坐标。默认 `candidates.json` 使用原图坐标，另存 ROI 偏移信息。

## Camera Calibration And Undistortion

参考资料：

- OpenCV camera calibration: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
- OpenCV Python calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

处理流程：

- 读取棋盘格图片目录。
- 根据 `board_cols`、`board_rows`、`square_size_mm` 生成 object points。
- 使用 `cv2.findChessboardCorners` 和 `cv2.cornerSubPix` 查找角点。
- 使用 `cv2.calibrateCamera` 估计 camera matrix、distortion coefficients、rvecs、tvecs。
- 使用 `cv2.getOptimalNewCameraMatrix` 和 `cv2.undistort` 输出校正图。
- 计算 reprojection error。
- 基于 `square_size_mm` 和图像中的角点间距，给出近似 pixel/mm 关系说明。

`calibration_result.yaml` 应包含：

- image_size
- board_size
- square_size_mm
- camera_matrix
- distortion_coefficients
- reprojection_error
- valid_images
- skipped_images
- approximate_pixel_per_mm

## Normal Sample Statistics

处理流程：

- 读取一组正常样本。
- 可选执行 resize、ROI、灰度化、光照归一化。
- 将样本堆叠为数组，计算 mean、variance、std。
- 对每张图计算 `abs(image - mean)` 的 diff map。
- 输出全局统计：平均 std、P95/P99 diff、最大 diff 区域。
- 生成 diff overlay，用于观察正常波动可能导致的误报区域。

如果样本存在位置偏移，本练习先通过 README 提醒风险，不强制实现配准；后续可以在单独变更中加入模板匹配或特征配准。

## Configuration

`sample_config.yaml` 建议包含：

```yaml
roi:
  x: 0
  y: 0
  width: null
  height: null
preprocess:
  grayscale: true
  blur_kernel: 3
  illumination:
    method: divide
    background_kernel: 51
threshold:
  method: otsu
  fixed_value: 40
morphology:
  open_kernel: 3
  close_kernel: 5
connected_components:
  min_area: 20
  max_area: null
  min_width: 2
  min_height: 2
calibration:
  board_cols: 9
  board_rows: 6
  square_size_mm: 25.0
normal_stats:
  resize_width: null
  resize_height: null
```

## Error Handling

- 输入路径不存在时给出明确错误。
- 图片无法读取时跳过并写入 summary。
- 标定角点不足时退出并提示需要更多有效棋盘格图片。
- ROI 超出图片边界时拒绝执行并显示图片尺寸。

## Testing Strategy

- 使用小型合成图片测试 baseline 的连通域输出。
- 使用少量可控正常样本测试 mean/variance/diff 输出是否存在。
- 标定脚本可先做 smoke test：无有效角点时应给出清晰失败信息；有数据后再验证 reprojection error 输出。

## Tradeoffs

- 选择传统 OpenCV 脚本而不是 notebook，便于命令行复现和后续纳入自动化测试。
- 光照归一化先实现常见方法，不追求覆盖所有工业光学场景。
- pixel/mm 关系先强调近似与条件，避免误导为全视野固定物理尺度；真实项目需要结合标定平面、工作距离和外参。
