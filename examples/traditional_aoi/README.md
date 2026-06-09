# Traditional AOI OpenCV Practice

这个目录对应路线图第 1-2 周的传统视觉练习，目标是先跑通一个可解释的 AOI baseline，再理解标定、畸变校正和正常样本波动。

## Environment

需要 Python 3.10+，并安装：

```bash
pip install opencv-python numpy pyyaml
```

## 1. Traditional AOI Baseline

输入可以是单张图片或图片目录：

```bash
python examples/traditional_aoi/traditional_aoi_baseline.py \
  --input data/aoi/image.png \
  --config examples/traditional_aoi/sample_config.yaml \
  --output outputs/traditional_aoi/baseline
```

流程是：读取图片 -> ROI -> 灰度/blur -> 光照归一化 -> 阈值/边缘 -> 形态学 -> 连通域 -> 输出候选区域。

主要输出：

- `outputs/traditional_aoi/baseline/candidates.json`：每张图的候选 bbox、area、centroid、mean intensity。bbox 默认是原图坐标，`bbox_roi` 是 ROI 内坐标。
- `outputs/traditional_aoi/baseline/masks/`：二值候选 mask。
- `outputs/traditional_aoi/baseline/overlays/`：带候选框的原图 overlay。
- `outputs/traditional_aoi/baseline/edges/`：Canny edge 辅助结果。

调参入口集中在 `sample_config.yaml`：ROI、光照归一化方式、阈值方式、Canny 参数、形态学核大小、连通域面积和形状过滤。

## 2. Camera Calibration And Undistortion

准备一组同一相机、同一分辨率拍摄的棋盘格图片，并配置棋盘格内角点数量和方格实际尺寸：

```bash
python examples/traditional_aoi/camera_calibration.py \
  --input data/calibration/chessboard \
  --config examples/traditional_aoi/sample_config.yaml \
  --output outputs/traditional_aoi/calibration
```

主要输出：

- `calibration_result.yaml`：image size、board size、square size、camera matrix、distortion coefficients、reprojection error、valid/skipped images、approximate pixel/mm。
- `undistorted/`：畸变校正后的图片。
- `comparison/`：校正前后横向对比图。

`approximate_pixel_per_mm` 只是在棋盘格所在平面、当前工作距离和当前成像条件下的近似值。真实 AOI 项目里，像素到物理尺寸的关系会受相机内参、外参、物体平面、镜头畸变、工作距离和透视影响；如果物体不在同一平面或相机姿态变化，不能把这个值当作全视野固定比例。

参考：

- OpenCV camera calibration: https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html
- OpenCV Python calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

## 3. Normal Sample Statistics

准备一组正常样本，建议来自同一工位、同一产品、相同曝光和光源设置：

```bash
python examples/traditional_aoi/normal_sample_statistics.py \
  --input data/aoi/normal \
  --config examples/traditional_aoi/sample_config.yaml \
  --output outputs/traditional_aoi/normal_stats
```

主要输出：

- `mean.png`：正常样本均值图。
- `variance.png`：正常样本方差图。
- `std.png`：正常样本标准差图。
- `diff_overlays/`：每张图与均值图的差异热力 overlay。
- `summary.json`：样本数、average std、P95/P99 diff、每张图的 mean/max abs diff。

看图时重点关注两件事：方差图亮的区域往往是正常波动更大的区域，直接套固定阈值容易误报；diff overlay 反复亮的位置通常是光照、定位、纹理或反光带来的正常变化，后续可以用更稳定的 ROI、配准、光学改造或区域化阈值来处理。

如果正常样本尺寸不一致，脚本会要求配置 `normal_stats.resize_width` 和 `normal_stats.resize_height`。如果样本存在明显位置偏移，先不要急着调阈值；这通常说明需要配准或更稳定的治具/触发。

## Smoke Test Ideas

可以用下面的小脚本生成极简合成图，再运行 baseline：

```bash
python - <<'PY'
from pathlib import Path
import cv2
import numpy as np

out = Path("outputs/traditional_aoi/smoke_inputs")
out.mkdir(parents=True, exist_ok=True)
img = np.full((160, 220, 3), 180, np.uint8)
cv2.circle(img, (90, 70), 12, (40, 40, 40), -1)
cv2.rectangle(img, (140, 95), (168, 118), (245, 245, 245), -1)
cv2.imwrite(str(out / "synthetic_defect.png"), img)
PY

python examples/traditional_aoi/traditional_aoi_baseline.py \
  --input outputs/traditional_aoi/smoke_inputs/synthetic_defect.png \
  --output outputs/traditional_aoi/baseline
```

正常样本统计也可以用几张轻微扰动的合成图做 smoke test：

```bash
python - <<'PY'
from pathlib import Path
import cv2
import numpy as np

out = Path("outputs/traditional_aoi/normal_smoke_inputs")
out.mkdir(parents=True, exist_ok=True)
for i, value in enumerate([120, 123, 126]):
    img = np.full((120, 160, 3), value, np.uint8)
    cv2.rectangle(img, (40 + i, 35), (90 + i, 80), (150, 150, 150), -1)
    cv2.imwrite(str(out / f"normal_{i}.png"), img)
PY

python examples/traditional_aoi/normal_sample_statistics.py \
  --input outputs/traditional_aoi/normal_smoke_inputs \
  --output outputs/traditional_aoi/normal_stats
```

## Verification

本次实现已执行语法检查：

```bash
./.venv/bin/python -m py_compile \
  examples/traditional_aoi/common.py \
  examples/traditional_aoi/traditional_aoi_baseline.py \
  examples/traditional_aoi/camera_calibration.py \
  examples/traditional_aoi/normal_sample_statistics.py
```

结果：通过。当前 `.venv` 未安装 `opencv-python`，所以依赖 OpenCV 的 smoke test 需要先安装环境依赖后再运行。
