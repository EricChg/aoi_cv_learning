# Tasks: Traditional AOI baseline practice

## 1. Project Setup

- [x] 1.1 新建 `examples/traditional_aoi/` 目录。
- [x] 1.2 新增 `sample_config.yaml`，包含 ROI、预处理、阈值、形态学、连通域、标定和正常样本统计参数。
- [x] 1.3 新增 `README.md`，说明三组练习的目标、输入数据要求、命令示例和输出解释。

## 2. Traditional AOI Baseline

- [x] 2.1 实现图片读取，支持单张图片和目录批处理。
- [x] 2.2 实现 ROI 裁剪，并保留 ROI 偏移用于坐标还原。
- [x] 2.3 实现灰度化、blur 和光照归一化。
- [x] 2.4 实现阈值、Canny edge、形态学处理和连通域分析。
- [x] 2.5 输出 mask、overlay 和 `candidates.json`。
- [x] 2.6 增加面积、宽高、长宽比过滤参数。

## 3. Camera Calibration

- [x] 3.1 实现棋盘格图片读取和角点检测。
- [x] 3.2 实现 object points 构造，支持配置棋盘格内角点数量和方格物理尺寸。
- [x] 3.3 调用 OpenCV calibrateCamera 计算内参、畸变参数和 reprojection error。
- [x] 3.4 输出 undistorted 图片、校正前后对比图和 `calibration_result.yaml`。
- [x] 3.5 在 README 中解释 pixel/mm 的近似条件和局限。

## 4. Normal Sample Statistics

- [x] 4.1 实现正常样本批量读取、ROI、灰度化和可选 resize。
- [x] 4.2 计算 mean map、variance map、std map。
- [x] 4.3 对每张正常图输出 diff map 和 diff overlay。
- [x] 4.4 输出 `summary.json`，包含样本数、平均 std、P95/P99 diff 等统计。
- [x] 4.5 在 README 中说明如何从方差图和差分图判断正常波动区域。

## 5. Verification

- [x] 5.1 为传统 AOI baseline 添加至少一个合成图 smoke test 或可复现命令。
- [x] 5.2 为正常样本统计添加至少一个小样本 smoke test 或可复现命令。
- [x] 5.3 运行格式检查或基础测试，记录命令和结果。
- [x] 5.4 确认输出目录结构与 README 一致。
