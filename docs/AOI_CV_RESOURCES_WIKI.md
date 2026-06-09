# AOI CV Resources Wiki

这个 wiki 用来持续整理 AOI/CV 学习资料、项目模板和工程检查清单。

## 目录

- [核心概念](#核心概念)
- [计算机视觉基础](#计算机视觉基础)
- [任务类型选型](#任务类型选型)
- [公开数据集](#公开数据集)
- [论文与方法](#论文与方法)
- [工具链](#工具链)
- [传统 AOI Baseline](#传统-aoi-baseline)
- [部署与服务](#部署与服务)
- [AOI 需求模板](#aoi-需求模板)
- [训练实验模板](#训练实验模板)
- [上线评审模板](#上线评审模板)

## 核心概念

### AOI

Automated Optical Inspection，自动光学检测。核心不是单个模型，而是相机、镜头、光源、采集、算法、判定、产线通信、数据闭环组成的系统。

### 缺陷检测

AOI 中的缺陷检测可以拆成：

- image-level：整张图 OK/NG。
- object-level：缺陷 bbox、类别、置信度。
- pixel-level：缺陷 mask、面积、形态。
- measurement-level：尺寸、位置、间距、角度等几何量。

### 异常检测

当缺陷样本少、缺陷类型未知、正常样本充分时，优先考虑视觉异常检测。常见输出是 anomaly score 与 heatmap，再通过阈值和后处理转成业务判定。

## 计算机视觉基础

这部分可以理解为：做 AOI/CV 项目时，除了会训练模型，还要懂图像本身、传统视觉处理、几何校正和图像质量控制。很多产线问题不是模型本身造成的，而是成像、光照、标定、定位出了问题。

### 要掌握

- 图像表示：灰度/RGB/HSV/Lab，bit depth，动态范围，gamma，白平衡。
- 基础算子：滤波、边缘、形态学、阈值、连通域、轮廓、霍夫变换。
- 几何：相机内参/外参、畸变校正、透视变换、模板匹配、图像配准。
- 质量控制：模糊、过曝、欠曝、反光、阴影、噪声、视野偏移。

### 图像表示

`灰度 / RGB / HSV / Lab` 是不同的图像表达方式。

`灰度图` 只有亮度信息，适合处理划伤、边缘、凹坑、脏污这类主要靠明暗变化识别的缺陷。

```text
一张灰度图中：
0   表示黑
255 表示白
中间值表示不同亮度
```

`RGB` 是红、绿、蓝三通道，适合保留原始颜色信息。例如检测印刷颜色错误、污染、色差时，RGB 比灰度更有用。

`HSV` 把颜色拆成色相、饱和度、亮度。它常用于颜色筛选，比如检测红色胶水、蓝色保护膜是否缺失。相比 RGB，HSV 对亮度变化有时更稳定。

```text
H：颜色类型，例如红、黄、蓝
S：颜色纯不纯
V：亮不亮
```

`Lab` 更接近人眼感知颜色的方式，常用于色差检测。

```text
L：亮度
a：绿到红
b：蓝到黄
```

如果要判断两个零件颜色是否偏色，可以在 Lab 空间里计算色差。

`bit depth` 是位深，表示每个像素能表达多少亮度层级。常见 8-bit 图像范围是 0 到 255；工业相机可能输出 10-bit、12-bit、16-bit，能保留更多细节。

```text
8-bit：0 ~ 255
12-bit：0 ~ 4095
16-bit：0 ~ 65535
```

对于微弱划伤、低对比度缺陷，高位深图像可能更容易保留细节。

`动态范围` 表示图像同时保留亮部和暗部细节的能力。动态范围不足时，亮处会过曝，暗处会死黑。

`gamma` 是亮度非线性变换。它会影响图像看起来的明暗分布。工业检测里要注意：训练图和上线图的 gamma 处理要一致，否则模型看到的亮度分布会变。

`白平衡` 用来校正颜色偏差。例如同一个白色产品，在不同光源下可能偏黄或偏蓝。白平衡不稳定会导致颜色类缺陷误判。

### 基础算子

`滤波` 用来降噪或增强图像。常见有均值滤波、高斯滤波、中值滤波。

例如图像里有很多盐粒状噪声，可以用中值滤波去掉小噪点；如果图像太锐利、有随机噪声，可以用高斯滤波平滑。

`边缘` 用来找亮暗变化明显的位置。常见算子有 Sobel、Canny。

例子：检测产品边框、孔位、裂纹边缘时，可以先提取边缘，再做几何测量。

`形态学` 常用于处理二值图，包括腐蚀、膨胀、开运算、闭运算。

```text
腐蚀：让白色区域变小，去掉小噪点
膨胀：让白色区域变大，连接断裂区域
开运算：先腐蚀再膨胀，去小噪声
闭运算：先膨胀再腐蚀，填小孔洞
```

例子：分割出胶水区域后，如果 mask 有小洞，可以用闭运算补齐。

`阈值` 是把灰度图变成黑白图。比如亮度大于 180 的像素认为是反光或白色缺陷。

```text
gray > 180 -> 缺陷候选
gray <= 180 -> 背景
```

如果光照不均，可以用自适应阈值或先做光照归一化。

`连通域` 是把二值图里相连的白色区域找出来。每个区域可以计算面积、宽高、中心点。

例子：检测异物时，可以先阈值分割，再用连通域过滤掉面积太小的噪声。

`轮廓` 是区域边界。可以用来计算面积、周长、外接矩形、最小包围圆、形状复杂度。

例子：判断一个毛刺是否超过尺寸阈值，可以提取轮廓后计算长度或面积。

`霍夫变换` 用来检测直线、圆等规则几何形状。

例子：检测圆孔位置是否偏移，可以用霍夫圆；检测产品边线是否倾斜，可以用霍夫直线。

### 几何

`相机内参` 描述相机自身成像特性，比如焦距、主点、像素比例。

`相机外参` 描述相机相对被测物或世界坐标的位置和姿态。

这些参数用于把图像坐标和真实物理坐标联系起来。

```text
图像坐标：像素，比如 x=500, y=300
物理坐标：毫米，比如 X=12.5mm, Y=8.0mm
```

`畸变校正` 用来修正镜头造成的图像变形。广角镜头或低成本镜头容易让直线变弯。做尺寸测量、孔位测量时，畸变必须处理。

`透视变换` 用来把倾斜拍摄的平面拉正。

例子：相机不是垂直拍 PCB，而是有角度，矩形区域在图里变成梯形。透视变换可以把它校正成俯视图，方便后续检测。

`模板匹配` 是用一个小模板去图中找最相似的位置。

例子：先找到产品上的定位孔、Logo 或角点，再根据它们定位 ROI。这样即使产品有轻微偏移，检测区域也能跟着移动。

`图像配准` 是把不同图片对齐。

比如同一个产品每次拍照位置略有偏移，可以先把当前图和标准图对齐，再做差分检测。

```text
当前图 -> 配准到标准图 -> 图像差分 -> 找异常区域
```

这在传统 AOI baseline 里非常常见。

### 质量控制

`模糊` 会让边缘变软、小缺陷消失。

比如细小划伤在清晰图里可见，但轻微失焦后就看不到了。可以用 Laplacian 方差等指标检测模糊。

`过曝` 是亮部细节丢失。

例如金属表面反光区域全变成白色，划伤和背景都变成 255，模型无法区分。

`欠曝` 是图像太暗，暗部细节丢失。

例如黑色塑料件上有脏污，但整体亮度太低，缺陷对比度不够。

`反光` 是 AOI 里很常见的问题，尤其是金属、玻璃、透明件。反光可能被误判成划伤、脏污、缺胶，也可能遮住真实缺陷。通常需要从光源角度、偏振片、曝光策略上解决。

`阴影` 会造成局部亮度变化，传统阈值和模型都可能受影响。夹具遮挡、产品翘曲、光源角度不合理都会产生阴影。

`噪声` 是图像里的随机干扰。曝光不足、增益过高、相机质量差时容易出现。噪声会导致误检，也会让边缘和阈值不稳定。

`视野偏移` 是产品在画面中的位置变化。

如果 ROI 是固定裁剪，产品稍微偏一点，缺陷区域可能被裁掉，或者背景进入检测区域。通常要用定位、模板匹配、机械限位或相机标定来控制。

一句话总结：

```text
图像表示：知道图像里存的是什么信息。
基础算子：知道怎么从图像里提取边缘、区域、形状。
几何：知道怎么定位、校正、测量。
质量控制：知道什么样的图不能直接拿来检测。
```

在 AOI 项目里，这些基础知识决定了你能不能判断问题到底来自模型、数据、光源、相机、夹具还是产线位置偏移。

## 任务类型选型

| 场景 | 首选方案 | 备注 |
| --- | --- | --- |
| 缺陷类别清楚且样本足够 | 检测/分割 | YOLO、Mask R-CNN、U-Net、SegFormer |
| 正常样本多，缺陷少 | 异常检测 | PatchCore、EfficientAD、PaDiM、STFPM |
| 缺陷极小 | 高分辨率 tile + 检测/分割/异常检测 | 注意坐标还原和 overlap |
| 缺陷表现为尺寸/位置异常 | 传统视觉 + 几何测量 | 配准、模板、边缘、轮廓 |
| 反光、透明、纹理强波动 | 先优化成像，再建模 | 光源和夹具可能比模型更关键 |
| 需要解释面积/长度 | 分割优先 | mask 后处理得到几何量 |
| 实时节拍很紧 | 轻量模型 + ROI + TensorRT/OpenVINO | 先减少输入和无效区域 |

## 公开数据集

### MVTec AD

- 链接：https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- 重点：工业异常检测经典数据集，15 个对象/纹理类别，超过 5000 张高分辨率图，提供像素级异常标注。
- 用法：入门异常检测、PatchCore/EfficientAD/PaDiM benchmark。
- 注意：许可为 CC BY-NC-SA 4.0，不可直接用于商业用途。

### MVTec AD 2

- 链接：https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2
- 重点：2026 年 IJCV 数据集，8 个更复杂的工业异常场景，超过 8000 张高分辨率图。
- 特点：小缺陷、大图、边缘缺陷、透明/反光物体、重叠物体、不同光照条件、正常状态高变化。
- 适合：进阶评估异常检测方法的鲁棒性。
- 注意：许可为 CC BY-NC-SA 4.0。

### MVTec 3D-AD

- 链接：https://www.mvtec.com/research-teaching/datasets
- 重点：3D 工业异常检测，适合学习深度/点云/表面形貌类 AOI。

### VisA

- 关键词：Visual Anomaly dataset, industrial anomaly detection。
- 重点：工业异常检测常用 benchmark，可与 MVTec AD 互补。

## 论文与方法

### PatchCore

- 论文页：https://www.amazon.science/publications/towards-total-recall-in-industrial-anomaly-detection
- 核心思想：用 ImageNet 预训练 backbone 提取 patch 特征，通过 memory bank 建模正常样本，测试时用最近邻距离得到异常分数和热力图。
- 适合：缺陷样本少、冷启动、需要定位异常区域的 AOI。
- 风险：对位置偏移、旋转、光照变化和正常模式多样性敏感，需要配准、增强和阈值策略。

### EfficientAD

- 论文页：https://huggingface.co/papers/2303.14535
- 核心思想：轻量特征提取 + student-teacher + autoencoder，全局和局部异常兼顾，强调毫秒级低延迟。
- 适合：实时 AOI、边缘部署、缺陷样本少的场景。

### PaDiM

- 关键词：Patch Distribution Modeling。
- 核心思想：对正常 patch 特征建高斯分布，用 Mahalanobis distance 检测异常。
- 适合：入门理解“正常特征分布”。

### STFPM

- 关键词：Student-Teacher Feature Pyramid Matching。
- 核心思想：student 学 teacher 的正常特征，多尺度特征差异作为异常信号。

### FastFlow / CFlow

- 关键词：normalizing flow anomaly detection。
- 核心思想：用 flow 建模正常特征分布，输出 likelihood 或 anomaly map。

### U-Net / DeepLab / SegFormer

- 用途：缺陷分割。
- 优势：输出 mask，便于计算面积、长度、边界。
- 条件：需要像素级标注，标注成本较高。

### YOLO

- 文档：https://docs.ultralytics.com/
- 用途：实时目标检测、分割、分类、OBB、tracking。
- 适合：缺陷类别明确、样本足够、节拍要求高的场景。
- 注意：Ultralytics 开源版本采用 AGPL-3.0，商业使用需要关注许可。

## 工具链

### OpenCV

- 文档：https://docs.opencv.org/4.x/
- 用途：图像处理、相机标定、配准、传统视觉 baseline、后处理。
- 必学模块：calib3d、imgproc、features2d、contours、threshold、morphology。

### PyTorch

- 迁移学习教程：https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- 用途：模型训练、微调、自定义 dataset、实验管理。

### Anomalib

- 文档：https://anomalib.readthedocs.io/en/v1/
- 用途：视觉异常检测训练、推理、benchmark。
- 价值：快速复现 PatchCore、PaDiM、STFPM、EfficientAD 等方法。

### Ultralytics YOLO

- 文档：https://docs.ultralytics.com/
- 用途：检测、分割、分类、导出 ONNX/TensorRT。
- 价值：快速训练 AOI 检测/分割 baseline。

### Labeling

- CVAT：适合团队标注、检测/分割任务。
- Label Studio：适合多模态与灵活标注。
- Roboflow：适合快速数据转换和数据集管理，但企业数据需注意合规。

### Experiment Tracking

- MLflow：模型与实验版本管理。
- Weights & Biases：实验可视化和团队协作。
- DVC：数据版本管理。

## 传统 AOI Baseline

这个 baseline 的目标不是“一步做到最好”，而是先建立一条可解释、可调参、能快速发现问题的传统视觉检测链路。它特别适合 AOI 早期：缺陷形态相对稳定、背景结构固定、相机位置固定、产品姿态变化不大。

整体流程：

```text
读取图片 -> ROI -> 光照归一化 -> 阈值/边缘 -> 形态学处理 -> 连通域 -> 缺陷候选区域
```

### 1. 读取图片

输入通常是工业相机采集到的原图。第一步要把图片读进来，并确认图像质量。

```python
img = cv2.imread("sample.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

要关注：

- 图像是否为空，路径或格式是否正确。
- 分辨率是否符合预期。
- 是否过曝、欠曝、模糊。
- 是否需要转灰度图。

AOI 里很多缺陷主要体现在亮度、纹理、边缘变化上，所以传统 baseline 通常先用灰度图做。如果缺陷是色差、污染、印刷颜色异常，就需要保留 RGB/HSV/Lab 等颜色空间。

### 2. ROI

ROI 是 Region of Interest，也就是只看真正需要检测的区域。

整张图里可能有夹具、背景、边框、螺丝、反光区域，这些区域不是检测对象，直接参与算法会造成误检。

固定裁剪示例：

```python
roi = gray[y1:y2, x1:x2]
```

如果产品位置稳定，可以手动配置 ROI。如果位置会轻微偏移，通常要先做定位：

```text
原图 -> 模板匹配/边缘定位/圆孔定位 -> 找到产品位置 -> 动态裁剪 ROI
```

ROI 的作用：

- 减少误检。
- 降低计算量。
- 让阈值更稳定。
- 方便后续把缺陷坐标映射回原图。

工程上要保存 ROI 配置：

```json
{
  "x": 120,
  "y": 80,
  "width": 900,
  "height": 600
}
```

### 3. 光照归一化

工业现场最常见的问题之一是光照不均：中间亮、边缘暗，或者有阴影、反光、批次表面亮度变化。如果直接阈值分割，算法可能把暗角、阴影当成缺陷。

常见做法包括：

**模糊背景扣除**

```python
blur = cv2.GaussianBlur(roi, (51, 51), 0)
norm = cv2.divide(roi, blur, scale=255)
```

含义是：用大尺度模糊估计“背景光照”，再把原图除以背景，让局部纹理和缺陷更突出。

**顶帽/黑帽变换**

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel)
```

- `TOPHAT`：突出比周围更亮的小区域。
- `BLACKHAT`：突出比周围更暗的小区域。

比如白色产品上的黑点，用 blackhat 往往有效；深色背景上的亮划痕，用 tophat 可能有效。

**CLAHE 局部直方图均衡**

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
norm = clahe.apply(roi)
```

CLAHE 能增强局部对比度，但也可能放大噪声，所以要谨慎调参。

### 4. 阈值

阈值的目标是把“疑似缺陷像素”从背景里分出来。

固定阈值：

```python
_, mask = cv2.threshold(norm, 40, 255, cv2.THRESH_BINARY)
```

适合光照和产品稳定的场景。

Otsu 自动阈值：

```python
_, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

适合前景和背景灰度分布比较明显的场景。

自适应阈值：

```python
mask = cv2.adaptiveThreshold(
    norm, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 5
)
```

适合局部光照不均，但也更容易产生碎片噪声。

在 AOI 里，阈值经常不是一个全局值，而是按产品型号、工位、ROI 或缺陷类型配置。

### 5. 边缘检测

有些缺陷不是明显的亮暗块，而是边缘异常，比如裂纹、划痕、缺口、毛刺。

```python
edges = cv2.Canny(norm, 50, 150)
```

Canny 会输出边缘像素。它适合找细长结构，但对噪声和纹理也敏感。

常见处理方式：

```text
高斯滤波 -> Canny -> 形态学闭运算 -> 连通域分析
```

示例：

```python
edges = cv2.GaussianBlur(norm, (5, 5), 0)
edges = cv2.Canny(edges, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

实际项目里，阈值和边缘经常会组合：

```text
暗点 mask + 亮点 mask + 边缘异常 mask -> merged mask
```

### 6. 形态学处理

阈值或边缘得到的 mask 往往很脏：有孤立噪点、缺陷断裂、小孔洞。形态学用于清理 mask。

可以把 `mask` 理解成一张黑白图：

- 白色区域：算法认为这里可能是缺陷、边缘或目标。
- 黑色区域：算法认为这里不是目标。

形态学操作不关心纹理、颜色和语义，只看白色区域的几何形状，然后做变大、变小、断开、连接、填洞等处理。

核心参数是 `kernel`，也叫结构元素。它可以理解成一个“小刷子”或“小探针”。OpenCV 会拿这个小窗口在整张 mask 上滑动，决定白色区域应该保留、扩大还是删除。

常见 kernel：

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
```

- `MORPH_RECT`：矩形，通用，适合方块状或方向性明显的结构。
- `MORPH_ELLIPSE`：椭圆/近似圆，适合点状、圆斑状缺陷。
- `(3, 3)`：轻微处理，细节保留多。
- `(9, 9)`：处理更强，噪声更少，但小缺陷可能被吃掉。
- `(15, 3)`：横向长条，适合连接横向划痕。
- `(3, 15)`：纵向长条，适合连接纵向划痕。

腐蚀：缩小白色区域。

```python
mask = cv2.erode(mask, kernel, iterations=1)
```

腐蚀会让白色区域往里缩。小白点可能直接消失，大区域会变瘦，细线可能断掉，两个勉强连在一起的区域可能被分开。

适合：

- 去掉细小白色噪声。
- 分开轻微粘连的区域。

风险：

- 真实的小缺陷也可能被吃掉。
- 细小划痕可能被腐蚀断裂。

膨胀：扩大白色区域。

```python
mask = cv2.dilate(mask, kernel, iterations=1)
```

膨胀会让白色区域向外扩。缺陷区域会变粗、变大，断开的线段可能连起来，小孔洞可能被填住，相邻区域也可能粘在一起。

适合：

- 连接断裂边缘。
- 补全划痕、裂纹等细长缺陷。
- 扩大候选区域，方便后续连通域分析。

风险：

- 噪声也会被放大。
- 两个不同缺陷可能被合并成一个。

开运算：去小噪点。

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```

开运算 = 先腐蚀，再膨胀。

直觉是：先用腐蚀把很小的白色噪点干掉，再用膨胀把剩下的大区域恢复得差不多。

典型场景：

```text
原 mask：   大缺陷 + 很多小白点噪声
开运算后： 大缺陷还在，小白点消失
```

注意：如果真实缺陷也很小，开运算可能把它一起删掉。小缺陷检测时，kernel 不能太大。

闭运算：连接断裂、填小洞。

```python
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

闭运算 = 先膨胀，再腐蚀。

直觉是：先用膨胀把断裂的地方连起来、把小洞填上，再用腐蚀把区域恢复得差不多。

典型场景：

```text
原 mask：   一条划痕被分成好几段
闭运算后： 几段划痕被连成一条
```

```text
原 mask：   缺陷区域中间有黑色小孔
闭运算后： 小孔被填平
```

调 kernel 很关键：

- 小 kernel：保留细节，但噪声多。
- 大 kernel：更干净，但可能吃掉小缺陷。
- 长条 kernel：适合连接划痕。
- 圆形/椭圆 kernel：适合点状缺陷。

AOI 调参时可以按缺陷形态来选：

```python
# 点状缺陷
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 横向划痕
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 纵向划痕
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

一句话总结：

- `erode`：白色变小，去小东西，但可能吃掉缺陷。
- `dilate`：白色变大，连接/补全，但可能放大噪声。
- `open`：先缩再放，主要去小噪点。
- `close`：先放再缩，主要连断裂、填小洞。
- `kernel`：决定处理力度和处理方向，需要结合缺陷尺寸、方向、形状来调。

### 7. 连通域分析

连通域就是把 mask 中相互连接的白色区域找出来，每一块都可能是一个缺陷候选。

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
```

`stats` 里会有：

```text
x, y, width, height, area
```

然后可以按规则过滤：

```python
candidates = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    if area < 20:
        continue
    if area > 5000:
        continue

    aspect = w / max(h, 1)
    if aspect > 20:
        defect_type = "scratch_like"
    else:
        defect_type = "spot_like"

    candidates.append((x, y, w, h, area, defect_type))
```

过滤规则通常包括：

- 面积太小：噪声。
- 面积太大：光照异常、背景区域、污染。
- 长宽比：划痕 vs 点状缺陷。
- 位置：某些区域允许结构边缘，不应判为缺陷。
- 形状：圆度、矩形度、细长程度。
- 灰度差：缺陷区域与周围背景差异是否足够。

### 8. 输出缺陷候选区域

最终输出不是只给 OK/NG，而是要保留可解释信息。

典型输出：

```json
{
  "result": "NG",
  "defects": [
    {
      "x": 341,
      "y": 128,
      "w": 26,
      "h": 8,
      "area": 92,
      "type": "scratch_like",
      "score": 0.76
    }
  ]
}
```

如果前面裁剪过 ROI，要把坐标映射回原图：

```python
global_x = roi_x + x
global_y = roi_y + y
```

也可以把候选区域画回原图：

```python
for x, y, w, h, area, defect_type in candidates:
    cv2.rectangle(
        img,
        (roi_x + x, roi_y + y),
        (roi_x + x + w, roi_y + y + h),
        (0, 0, 255),
        2
    )
```

### 完整 baseline 伪代码

```python
import cv2

img = cv2.imread("sample.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

roi_x, roi_y, roi_w, roi_h = 120, 80, 900, 600
roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

blur = cv2.GaussianBlur(roi, (51, 51), 0)
norm = cv2.divide(roi, blur, scale=255)

_, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

defects = []

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]

    if area < 20:
        continue
    if area > 5000:
        continue

    defects.append({
        "x": roi_x + int(x),
        "y": roi_y + int(y),
        "w": int(w),
        "h": int(h),
        "area": int(area)
    })

result = "NG" if defects else "OK"
```

一句话总结：这条传统 AOI baseline 的核心是，先通过 ROI 和光照归一化把问题变简单，再用阈值/边缘把可疑像素提出来，最后用连通域和规则把像素级结果变成业务可理解的缺陷候选。

## 部署与服务

### ONNX Runtime

- 性能调优：https://onnxruntime.ai/docs/performance/tune-performance/
- 适合：跨平台推理、CPU/GPU 部署、快速服务化。
- 关注点：execution provider、thread、I/O binding、graph optimization。

### TensorRT

- 文档：https://docs.nvidia.com/tensorrt/index.html
- 量化文档：https://docs.nvidia.com/deeplearning/tensorrt/10.13.2/inference-library/work-quantized-types.html
- 适合：NVIDIA GPU/Jetson，追求低延迟和高吞吐。
- 关注点：dynamic shape、FP16、INT8、校准集、plugin 算子。

### OpenVINO

- Model Server：https://docs.openvino.ai/2025/openvino-workflow/model-server/ovms_what_is_openvino_model_server.html
- 适合：Intel CPU/iGPU/NPU 边缘部署。
- 关注点：模型转换、INT8 量化、REST/gRPC serving。

### 服务架构关键词

- Model Registry：模型文件、版本、配置、指标、回滚。
- Config Center：阈值、ROI、后处理、产品型号配置。
- Result Store：原图、输入图、预测结果、人工复判、元数据。
- Online Monitor：延迟、吞吐、错误率、误报率、复判率、漂移。
- Data Loop：hard negative、低置信样本、人工复判样本进入再训练。

## AOI 需求模板

````markdown
# AOI 需求澄清

## 需求表

| 模块 | 字段 | 需填写内容 | 备注 |
| --- | --- | --- | --- |
| 业务背景 | 产品/工序 |  | 产品型号、检测工站、当前人工或规则方案 |
| 检测对象 | 材质/尺寸/检测面 |  | 材质、外形尺寸、需要检测的面或 ROI |
| 检测对象 | 运动与成像 |  | 静止/运动拍照，相机、镜头、光源、曝光 |
| 缺陷定义 | 缺陷类别 |  | 划伤、脏污、异物、缺料、错位、毛刺、裂纹、色差等 |
| 缺陷定义 | 最小可检缺陷 |  | 最小长度、面积、颜色差、位置偏差等 |
| 缺陷定义 | OK/NG 规则 |  | 哪些缺陷必须 NG，哪些进入复判或可接受 |
| 输出形态 | 判定与定位 |  | OK/NG、类别、bbox、mask、中心点、轮廓 |
| 输出形态 | 测量与追溯 |  | 面积、长度、等级、原图、结果图、SN、时间戳 |
| 指标 | 质量指标 |  | 漏检率、误检率、召回率、mAP、IoU、AUROC、PRO |
| 指标 | 性能指标 |  | 节拍、延迟上限、吞吐、复判率上限 |
| 数据 | 样本数量 |  | 正常样本、缺陷样本、类别分布、批次跨度 |
| 数据 | 标注与历史数据 |  | 图像级、框级、像素级；历史误判图、复判日志 |
| 部署 | 硬件与位置 |  | 工控机、CPU/GPU/NPU、边缘端/服务器/云端 |
| 部署 | 系统对接 |  | PLC、MES、HMI、数据库、日志、模型更新方式 |

## 指标说明

这些指标是 AOI 项目里的核心验收指标。它们不是单纯的算法指标，而是决定系统能不能上产线、能不能替代人工、能不能和设备节拍配合的业务指标。

### 漏检率目标

漏检就是：真实有缺陷，但系统判成 OK。

这是 AOI 里通常最危险的指标，因为漏检会让不良品流到下一道工序，甚至流到客户手里。

```text
漏检率 = 漏检的 NG 数量 / 实际 NG 总数量
```

例子：

```text
实际有 1000 个不良品
系统漏掉了 2 个
漏检率 = 2 / 1000 = 0.2%
```

填写时不要只写“越低越好”，要写成明确目标：

```markdown
- 漏检率目标：关键缺陷 <= 0.1%，一般缺陷 <= 0.5%
```

AOI 里经常按缺陷等级拆目标：

- 致命缺陷：漏检率必须极低。
- 主要缺陷：允许极少量漏检。
- 轻微缺陷：可以进入复判或抽检。

### 误检率目标

误检就是：真实是 OK，但系统判成 NG。

误检不会直接放走坏品，但会造成良品被拦下、人工复判增加、产线节拍变慢。误检率太高，现场会觉得系统“不好用”。

```text
误检率 = 被误判为 NG 的 OK 数量 / 实际 OK 总数量
```

例子：

```text
实际有 10000 个良品
系统误判了 150 个为 NG
误检率 = 150 / 10000 = 1.5%
```

填写示例：

```markdown
- 误检率目标：<= 2%
```

漏检率和误检率经常互相拉扯：阈值调严一点，漏检少了，但误检可能变多；阈值调松一点，误检少了，但漏检可能变多。

### 节拍

节拍指产线处理一个产品允许花多长时间，或者每隔多久会来一个产品。

它是业务/设备侧指标，不只是算法推理时间。

例子：

```text
产线每 1 秒过 1 个产品
AOI 系统必须在 1 秒内完成拍照、推理、后处理、通信和判定
```

填写示例：

```markdown
- 节拍：1 件 / 秒
- 节拍：单件总检测时间 <= 1000 ms
- 节拍：单件 4 张图，总处理时间 <= 800 ms
```

### 延迟上限

延迟上限指从输入到输出结果，最多允许多久。

它和节拍相关，但不是完全一回事。节拍更像“产线频率”，延迟更像“这件产品从拍照到拿到结果用了多久”。

延迟通常包括：

```text
图像采集
图像传输
预处理
模型推理
后处理
结果保存
PLC/MES 通信
```

比如模型推理只要 50 ms，但保存图片、网络通信、后处理加起来可能变成 500 ms。

填写示例：

```markdown
- 延迟上限：从相机触发到 PLC 收到 OK/NG <= 500 ms
```

这个定义要具体，否则后面容易扯皮：到底是只算模型推理，还是算完整链路。

### 复判率上限

复判率指系统没有直接给最终 OK/NG，而是把样本交给人工确认的比例。

常见进入复判的原因：

- 置信度不够。
- 疑似缺陷很小。
- 缺陷位于边界区域。
- 图像质量异常。
- 规则冲突。

```text
复判率 = 进入人工复判的数量 / 总检测数量
```

例子：

```text
一天检测 100000 件
其中 3000 件进入人工复判
复判率 = 3000 / 100000 = 3%
```

填写示例：

```markdown
- 复判率上限：<= 3%
```

复判率太高，说明系统虽然可能很谨慎，但没有真正替代人工。产线会觉得它只是把人工目检换成了人工复判。

比较完整的填写方式：

```markdown
## 指标
- 漏检率目标：关键缺陷 <= 0.1%，一般缺陷 <= 0.5%
- 误检率目标：<= 2%
- 节拍：单件 4 张图，总检测时间 <= 800 ms
- 延迟上限：从相机触发到 PLC 收到判定结果 <= 1000 ms
- 复判率上限：<= 3%
```

一句话总结：

```text
漏检率：坏品别放过。
误检率：良品别乱杀。
节拍：跟得上产线。
延迟上限：结果来得及用于拦截。
复判率：别把压力又丢回人工。
```

## 示例

| 模块 | 字段 | 示例内容 | 备注 |
| --- | --- | --- | --- |
| 业务背景 | 产品/工序 | 手机中框 CNC 后外观检测 | 目标替代人工目检 |
| 检测对象 | 材质/尺寸/检测面 | 阳极氧化铝；检测外侧四边和倒角区域 | 深灰/银色两种颜色 |
| 检测对象 | 运动与成像 | 转盘工位静止拍照；4 台 500 万像素相机 | PLC 硬触发 |
| 缺陷定义 | 缺陷类别 | 划伤、磕碰、脏污、毛刺 | 首期不做色差 |
| 缺陷定义 | 最小可检缺陷 | 划伤长度 >= 0.3 mm；脏污面积 >= 0.2 mm² | 低于阈值不计 NG |
| 缺陷定义 | OK/NG 规则 | 关键区域划伤/磕碰判 NG；疑似脏污进入复判 | 关键区域需标图 |
| 输出形态 | 判定与定位 | 输出 OK/NG、类别、置信度、bbox、结果叠图 | MES 接收最终判定 |
| 输出形态 | 测量与追溯 | 保存原图、结果图、SN、工位号、模型版本 | 保存 90 天 |
| 指标 | 质量指标 | 关键缺陷漏检率 <= 0.1%，误检率 <= 2% | 以人工复判为准 |
| 指标 | 性能指标 | 单件 4 张图总处理时间 <= 800 ms | 含前后处理 |
| 数据 | 样本数量 | 正常图 12000 张；缺陷图 2550 张 | 毛刺样本偏少 |
| 数据 | 标注与历史数据 | bbox 标注；有历史误判图 500 张 | 可做 hard negative |
| 部署 | 硬件与位置 | i7 工控机 + RTX 4060，本地边缘端部署 | 不依赖公网 |
| 部署 | 系统对接 | PLC 拦截信号、MES 上传结果、HMI 展示 NG 图 | 需定义接口协议 |
````

## 训练实验模板

### 任务字段说明

`任务：classification/detection/segmentation/anomaly` 用来说明这次实验解决的是哪一种视觉问题。AOI 项目里不要只写“缺陷检测”，因为不同任务对应的数据标注、模型输出、评估指标和上线方式都不一样。

- classification：图像级分类。输入一张图或一个 ROI，输出 OK/NG 或缺陷类别。适合只需要整图判定、不需要定位缺陷位置的场景，例如“这张图是否有脏污”。标注成本最低，常用指标包括 accuracy、recall、precision、F1、漏检率、误检率。
- detection：目标检测。输出缺陷类别、bbox 和置信度。适合缺陷位置需要展示或复判，但不强制精确边界的场景，例如划伤、磕碰、异物的大致位置。需要框标注，常用指标包括 mAP、每类 recall、每类 false positive、漏检样例分析。
- segmentation：语义/实例分割。输出像素级 mask。适合需要计算面积、长度、边界、覆盖率，或缺陷形态会影响判定的场景，例如胶水溢出、涂层缺失、裂纹长度。标注成本最高，常用指标包括 IoU、Dice、pixel-level recall、面积误差。
- anomaly：异常检测。通常用正常样本建模，输出 anomaly score、heatmap 或异常区域。适合缺陷样本少、缺陷类型未知、冷启动阶段，或产品正常状态比较稳定的场景。常用指标包括 image-level AUROC、pixel-level AUROC、PRO、阈值下的漏检率/误检率。

### 常用指标示例

#### Detection 指标

`mAP` 是 mean Average Precision，平均精度均值。可以理解为模型在“框得准不准、分类对不对、置信度排序好不好”上的综合分数。

例如检测手机外壳缺陷，有三类：划伤、磕碰、异物。模型预测了很多框，每个框都有类别和置信度：

```text
划伤 bbox 置信度 0.92
异物 bbox 置信度 0.81
磕碰 bbox 置信度 0.63
```

如果预测框和人工标注框重叠足够高，例如 IoU >= 0.5，并且类别也正确，就算一次正确检测。分别算出每一类的 AP，再取平均，就是 mAP。

```text
划伤 AP = 0.88
磕碰 AP = 0.76
异物 AP = 0.82

mAP = (0.88 + 0.76 + 0.82) / 3 = 0.82
```

`每类 recall` 表示每一类缺陷被找出来的比例，重点看漏检。例如测试集中真实有 100 个划伤，模型找出了 92 个：

```text
划伤 recall = 92 / 100 = 92%
```

如果真实有 50 个磕碰，只找出 35 个：

```text
磕碰 recall = 35 / 50 = 70%
```

这说明模型对磕碰容易漏检。

`每类 false positive` 表示每一类误报数量，也就是模型说“这里有缺陷”，但人工标注里其实没有。例如一批 1000 张良品图中，模型报出了 30 个异物缺陷：

```text
异物 false positive = 30
```

在产线里，这会导致良品被误判为 NG，增加复检压力。

`漏检样例分析` 不是单一数值，而是把模型没检出的真实缺陷拿出来看原因。

```text
漏检 20 个划伤：
- 8 个是划伤太浅，对比度低
- 5 个在强反光区域
- 4 个长度太短
- 3 个被产品边缘纹理干扰
```

这种分析能指导后续补数据、改光源、改标注规则或调阈值。

#### Segmentation 指标

`IoU` 是 Intersection over Union，交并比，用来衡量预测 mask 和真实 mask 的重叠程度。例如真实胶水溢出区域面积是 100 像素，模型预测区域是 120 像素，两者重叠 80 像素：

```text
IoU = 重叠面积 / 合并面积
    = 80 / (100 + 120 - 80)
    = 80 / 140
    = 57.1%
```

IoU 越高，说明 mask 越贴近真实缺陷边界。

`Dice` 是另一种衡量 mask 重叠的指标。相比 IoU，Dice 对小目标有时更敏感一些。还是上面的例子：

```text
Dice = 2 * 重叠面积 / (真实面积 + 预测面积)
     = 2 * 80 / (100 + 120)
     = 160 / 220
     = 72.7%
```

`pixel-level recall` 表示像素级召回率，即真实缺陷区域里有多少像素被模型找出来。例如真实裂纹有 1000 个像素，模型覆盖了其中 850 个：

```text
pixel-level recall = 850 / 1000 = 85%
```

如果这个指标低，说明模型把缺陷区域切碎了，或者漏掉了一部分，尤其对裂纹、涂层缺失这类形态敏感的缺陷很重要。

`面积误差` 表示预测缺陷面积和真实面积的差距，适合有面积判定规则的场景。例如真实胶水溢出面积是 2.0 mm²，模型预测为 2.4 mm²：

```text
面积误差 = |2.4 - 2.0| / 2.0 = 20%
```

如果业务规则是“溢胶面积超过 2.2 mm² 判 NG”，面积误差会直接影响最终判定。

#### Anomaly 指标

`image-level AUROC` 表示图像级异常区分能力，看模型能不能把整张图分成正常/异常。例如有 1000 张图，其中 800 张正常，200 张异常。模型给每张图输出一个 anomaly score：

```text
正常图 score 通常在 0.05 ~ 0.30
异常图 score 通常在 0.55 ~ 0.95
```

如果异常图的分数通常高于正常图，image-level AUROC 就高。这个指标适合判断“这张图要不要拦下来”。

`pixel-level AUROC` 表示像素级异常区分能力，看 heatmap 上的异常区域是否能对准真实缺陷位置。例如模型输出一张热力图，裂纹区域分数高，背景区域分数低。如果真实裂纹像素普遍比正常背景像素分数高，pixel-level AUROC 就高。这个指标适合判断“异常位置有没有找对”。

`PRO` 是 Per-Region Overlap，区域级重叠指标，常用于异常检测分割，尤其关注每个异常区域是否被覆盖，而不只是总体像素表现。例如一张图上有 3 个小异物，模型准确覆盖了大异物，但漏掉两个小异物。pixel-level AUROC 可能还不错，因为大区域贡献很多像素；但 PRO 会更明显地反映“小异常区域没覆盖”的问题。

`阈值下的漏检率/误检率` 是实际产线最常用的指标。因为上线时通常要设一个固定阈值，例如 anomaly score > 0.6 判 NG。

```text
阈值 = 0.6

真实异常 200 张：
- 检出 190 张
- 漏检 10 张

漏检率 = 10 / 200 = 5%

真实正常 800 张：
- 误报 40 张

误检率 = 40 / 800 = 5%
```

在 AOI 项目里，最终经常不是单看 AUROC，而是看某个业务阈值下能不能接受：

```text
漏检率 <= 1%
误检率 <= 5%
节拍 <= 200ms/张
```

简单总结：

```text
detection 看：框有没有找对、类别对不对
segmentation 看：缺陷区域边界和面积准不准
anomaly 看：异常能不能被区分出来，尤其适合缺陷少或类型未知
```

如果是产线验收，最关键的通常是：漏检率、误检率、每类 recall，再配合 mAP、IoU、AUROC 这些模型指标做技术评估。

选择建议：

```text
只要 OK/NG，不关心位置 -> classification
要知道缺陷大概在哪里 -> detection
要精确面积/长度/形态 -> segmentation
缺陷少、未知缺陷多、正常样本多 -> anomaly
```

```markdown
# Experiment Record

## 基本信息
- 实验 ID：
- 日期：
- 任务：classification / detection / segmentation / anomaly
- 数据版本：
- 代码版本：
- 模型版本：

## 数据
- train/val/test 切分：
- 类别分布：
- 标注质量问题：
- hard cases：

## 模型
- backbone：
- 输入尺寸：
- tile/ROI：
- loss：
- augmentation：
- optimizer：
- lr schedule：

## 结果
- image-level 指标：
- object/pixel-level 指标：
- 每类指标：
- latency：
- throughput：
- 失败样例：

## 结论
- 是否优于 baseline：
- 主要收益：
- 主要风险：
- 下一步：
```

## 上线评审模板

```markdown
# AOI Model Release Review

## Release
- 模型版本：
- 配置版本：
- 数据版本：
- 服务版本：

## Offline Metrics
- Recall：
- False positive rate：
- mAP/IoU/AUROC/PRO：
- 每类缺陷表现：

## Runtime
- 硬件：
- P50/P95/P99 latency：
- throughput：
- GPU/CPU/memory：

## Error Analysis
- 漏检样例：
- 误检样例：
- 光照/位置/批次敏感性：
- 未覆盖缺陷：

## Deployment Plan
- 灰度范围：
- 阈值配置：
- 回滚方案：
- 人工复判流程：
- 数据回流策略：

## Go/No-Go
- 结论：
- 必须修复：
- 可接受风险：
```
