# AOI CV Model Pipelines

这份文档详细展开 `AOI 分类模型与 Pipeline` 和 `AOI 检测分割异常与混合 Pipeline` 中提到的模型与方案，重点说明：

- 模型思想。
- 优点和缺点。
- 输入输出结构。
- 典型 pipeline。
- 性能、延迟和部署取舍。

## 1. AOI 分类模型

分类模型回答 image-level 问题：一张图是 `OK` 还是 `NG`，或者属于哪一种缺陷类别。

典型输入输出：

```text
输入：
image: [B, C, H, W]

输出：
single-label:
  logits: [B, num_classes]
  probs: softmax(logits)

multi-label:
  logits: [B, num_classes]
  probs: sigmoid(logits)
```

分类适合整图判定，但不直接给缺陷位置、面积和边界。

### 1.1 ResNet

ResNet 的核心思想是 residual connection。普通深层网络容易退化，层数增加后训练误差反而变高。ResNet 让每个 block 学习残差：

```text
output = F(x) + x
```

这样梯度更容易传递，深层 CNN 更稳定。

| 维度 | 说明 |
| --- | --- |
| 典型模型 | ResNet18、ResNet34、ResNet50、ResNet101 |
| 输入 | RGB 或灰度转 3 通道图像，常见尺寸 224x224、256x256、512x512 |
| 输出 | 分类 logits、类别概率 |
| 优点 | 稳定、资料多、预训练权重丰富、部署友好 |
| 缺点 | 参数效率和精度不一定优于现代 CNN；对极小缺陷需要高分辨率或 tile |
| AOI 适合场景 | 第一个分类 baseline、数据规模中小、需要稳定可解释的工程起点 |

ResNet 分类 pipeline：

```text
原图
 -> 图像质量检查
 -> ROI 裁剪 / 配准
 -> resize / normalize
 -> ResNet
 -> softmax / sigmoid
 -> 阈值判定
 -> OK/NG 或缺陷类别
```

性能特点：

- ResNet18 速度快，适合轻量 baseline。
- ResNet50 精度和速度较平衡，是常用起点。
- ResNet101 更重，精度提升不一定抵消延迟增加。
- ONNX、TensorRT、OpenVINO 支持成熟。

### 1.2 EfficientNet

EfficientNet 的核心思想是 compound scaling：同时按比例放大网络深度、宽度和输入分辨率，而不是只加深或只加宽。

```text
depth    -> 网络更深
width    -> 通道更多
resolution -> 输入更大
```

| 维度 | 说明 |
| --- | --- |
| 典型模型 | EfficientNet-B0 到 B7，AOI 常从 B0/B1/B2 开始 |
| 输入 | 常见 224x224 到 380x380，不同版本有推荐输入尺寸 |
| 输出 | 分类 logits、类别概率 |
| 优点 | 参数效率高，精度/速度比好 |
| 缺点 | 对输入尺寸、增强和训练策略敏感；部分算子在边缘部署上需验证 |
| AOI 适合场景 | 算力受限、边缘设备、需要比 ResNet 更高效率的分类任务 |

EfficientNet 分类 pipeline：

```text
原图
 -> ROI / tile
 -> 按模型推荐尺寸 resize
 -> normalize
 -> EfficientNet
 -> 类别概率
 -> per-class threshold
 -> OK/NG
```

性能特点：

- B0/B1 适合低延迟场景。
- B3 以上精度可能更好，但输入尺寸和延迟明显增加。
- 对小缺陷，提升输入分辨率可能比换更大模型更有效。
- 部署前要实际 benchmark，因为 depthwise convolution 在不同硬件上速度差异明显。

### 1.3 ConvNeXt

ConvNeXt 是现代化 CNN。它保留 CNN 的局部归纳偏置，同时吸收了 Transformer 时代的设计经验，例如更大的 kernel、更简洁的 block、LayerNorm、GELU 和现代训练策略。

| 维度 | 说明 |
| --- | --- |
| 典型模型 | ConvNeXt-Tiny、Small、Base |
| 输入 | 常见 224x224、384x384，也可用于 tile 输入 |
| 输出 | 分类 logits、类别概率 |
| 优点 | 表达能力强，纹理和局部结构建模好，工业分类表现稳定 |
| 缺点 | 比 ResNet18/50 更重，边缘端需要测试延迟 |
| AOI 适合场景 | 纹理缺陷、表面缺陷、希望提升分类 baseline 上限 |

ConvNeXt 分类 pipeline：

```text
原图
 -> 质量检查
 -> ROI / tile
 -> resize / normalize
 -> ConvNeXt
 -> 分类分数
 -> 阈值 + hard negative 过滤
 -> 判定
```

性能特点：

- ConvNeXt-Tiny 通常是较好的起点。
- 对工业纹理图像常比经典 ResNet 更有竞争力。
- 部署可走 ONNX / TensorRT，但要验证 LayerNorm 等算子支持和融合效果。

### 1.4 ViT

ViT 把图像切成 patch，把每个 patch 当成一个 token，再用 Transformer 做全局建模。

```text
image -> patches -> patch embeddings -> Transformer encoder -> class token -> classifier
```

| 维度 | 说明 |
| --- | --- |
| 典型模型 | ViT-B/16、ViT-L/16、DeiT、Swin Transformer |
| 输入 | 固定 patch size，例如 16x16；输入尺寸影响 token 数 |
| 输出 | 分类 logits、类别概率 |
| 优点 | 全局关系建模强，适合装配关系、排布关系、全局结构异常 |
| 缺点 | 小数据容易过拟合；训练和预训练依赖更强；高分辨率 token 数增长快 |
| AOI 适合场景 | 复杂全局结构、装配检查、强预训练模型可用的场景 |

ViT 分类 pipeline：

```text
原图
 -> 对齐 / ROI
 -> resize
 -> patch embedding
 -> Transformer encoder
 -> 分类头
 -> 阈值判定
```

性能特点：

- 没有足够数据时，不建议从零训练。
- 强依赖 ImageNet、MAE、DINO、CLIP 等预训练权重。
- 高分辨率输入会显著增加 token 数和计算量。
- Swin Transformer 这类层次化 Transformer 更适合高分辨率视觉任务。

## 2. 目标检测模型

目标检测回答 object-level 问题：缺陷在哪里、类别是什么、置信度是多少。

典型输入输出：

```text
输入：
image: [B, C, H, W]

输出：
detections:
  bbox: [x1, y1, x2, y2]
  class_id
  score
```

目标检测适合复判展示和缺陷定位，但 bbox 不是精确边界。

### 2.1 YOLO

YOLO 是单阶段实时检测模型。它直接在特征图上预测 bbox、类别和置信度。

| 维度 | 说明 |
| --- | --- |
| 输入 | 固定尺寸图像，例如 640x640、1024x1024 |
| 输出 | bbox、class、score，经过 NMS 得到最终检测框 |
| 优点 | 快、工程成熟、训练部署简单、适合实时产线 |
| 缺点 | 极小缺陷需要 tile、高分辨率输入或多尺度策略 |
| AOI 适合场景 | 划伤、异物、脏污、缺料、磕碰等需要位置但不要求精确边界的缺陷 |

YOLO pipeline：

```text
原图
 -> ROI / tile
 -> letterbox resize
 -> YOLO 推理
 -> NMS
 -> bbox 坐标还原
 -> 置信度 / 面积 / 位置规则过滤
 -> 输出缺陷列表
```

性能特点：

- 延迟低，适合 GPU、边缘 GPU 和部分 CPU 场景。
- 输入尺寸越大，小缺陷召回越好，但延迟越高。
- tile 可以提升小缺陷召回，但会增加推理次数。
- 实际上线要统计 P50/P95/P99 latency，而不是只看单张平均耗时。

### 2.2 Faster R-CNN

Faster R-CNN 是两阶段检测模型：第一阶段生成候选框，第二阶段对候选框分类和回归。

| 维度 | 说明 |
| --- | --- |
| 输入 | 图像，可配合 FPN 做多尺度特征 |
| 输出 | bbox、class、score |
| 优点 | 精度稳定，小数据或复杂背景下有时更稳 |
| 缺点 | 推理慢，部署复杂度高，不如 YOLO 适合强实时场景 |
| AOI 适合场景 | 节拍不紧、离线检测、精度优先、复杂背景缺陷 |

Faster R-CNN pipeline：

```text
原图 / ROI
 -> backbone + FPN
 -> RPN 生成候选框
 -> ROI Align
 -> 分类与 bbox 回归
 -> NMS
 -> 输出检测结果
```

性能特点：

- 对小目标可通过 FPN、anchor 设置和高分辨率输入改善。
- 延迟通常高于 YOLO。
- 更适合服务器或离线复检，不一定适合高节拍产线。

### 2.3 RT-DETR

RT-DETR 是实时 DETR 系列检测模型，用 Transformer 思路做端到端目标检测。

| 维度 | 说明 |
| --- | --- |
| 输入 | 图像，多尺度特征 |
| 输出 | 一组 object queries 对应的 bbox、class、score |
| 优点 | 全局建模强，端到端检测，减少传统 anchor/NMS 设计依赖 |
| 缺点 | 训练和部署链路相对复杂，对数据和调参更敏感 |
| AOI 适合场景 | 复杂背景、多目标关系、遮挡或希望尝试 Transformer 检测器的场景 |

RT-DETR pipeline：

```text
原图
 -> backbone 提取多尺度特征
 -> Transformer encoder / decoder
 -> object queries
 -> bbox + class
 -> 阈值过滤
 -> 坐标还原和业务规则
```

性能特点：

- 速度比传统 DETR 更适合实时场景，但仍需实际硬件 benchmark。
- 对全局上下文更友好。
- 在 AOI 中通常不是第一个 baseline，建议先用 YOLO 建立参照。

## 3. 分割模型

分割回答 pixel-level 问题：哪些像素属于缺陷。它适合面积、长度、边界和形态判定。

典型输入输出：

```text
输入：
image: [B, C, H, W]

语义分割输出：
mask_logits: [B, num_classes, H, W]

实例分割输出：
instances:
  bbox
  class_id
  score
  mask: [H, W]
```

### 3.1 U-Net

U-Net 是编码器-解码器结构，通过 skip connection 把低层细节传到解码阶段。

| 维度 | 说明 |
| --- | --- |
| 输入 | ROI 或 tile 图像 |
| 输出 | 像素级 mask |
| 优点 | 小数据友好，结构清晰，适合二分类缺陷分割 |
| 缺点 | 全局上下文能力有限，对复杂场景需要更强 backbone |
| AOI 适合场景 | 胶水、污染、裂纹、涂层缺失、表面缺陷 mask |

U-Net pipeline：

```text
原图
 -> ROI / tile
 -> U-Net
 -> mask probability
 -> 阈值二值化
 -> 形态学处理
 -> 连通域分析
 -> 面积 / 长度 / 位置判定
```

性能特点：

- 轻量版本速度快，适合边缘部署。
- 输入分辨率越高，mask 细节越好，但显存和延迟越高。
- 对极细裂纹可以结合 Dice loss、Focal loss、边界 loss。

### 3.2 DeepLab

DeepLab 使用空洞卷积和多尺度上下文模块，增强不同尺度区域的语义理解。

| 维度 | 说明 |
| --- | --- |
| 输入 | ROI 或 tile 图像 |
| 输出 | 语义分割 mask |
| 优点 | 多尺度上下文强，适合区域性缺陷 |
| 缺点 | 对极细边界可能需要额外后处理 |
| AOI 适合场景 | 涂层缺失、污染扩散、大面积异常区域 |

DeepLab pipeline：

```text
原图 / ROI
 -> resize
 -> DeepLab
 -> class probability map
 -> mask 后处理
 -> 几何量计算
 -> 业务阈值判定
```

性能特点：

- 比简单 U-Net 更重。
- 对大面积区域分割稳定。
- 需要验证输出 stride 对小缺陷边界的影响。

### 3.3 Mask R-CNN

Mask R-CNN 是实例分割模型，在 Faster R-CNN 的基础上为每个候选实例预测 mask。

| 维度 | 说明 |
| --- | --- |
| 输入 | 图像 |
| 输出 | bbox、class、score、instance mask |
| 优点 | 能区分多个缺陷实例，输出实例级 mask |
| 缺点 | 标注、训练、部署复杂，推理较慢 |
| AOI 适合场景 | 需要统计缺陷个数、每个缺陷面积和实例边界 |

Mask R-CNN pipeline：

```text
原图
 -> backbone + FPN
 -> RPN
 -> ROI Align
 -> 分类 / bbox / mask head
 -> instance mask
 -> 每个实例计算面积和位置
 -> 业务判定
```

性能特点：

- 比语义分割更适合多个缺陷实例分离。
- 延迟一般高于 YOLO 和 U-Net。
- 适合复检或精细分析，不一定适合高速产线。

### 3.4 SAM 辅助标注

SAM 是 promptable segmentation 模型。它可以根据点、框、粗 mask 等提示生成分割结果。

| 维度 | 说明 |
| --- | --- |
| 输入 | 图像 + prompt，例如点、框、文本或已有 mask |
| 输出 | 候选 mask |
| 优点 | 能显著提升标注效率 |
| 缺点 | 不是直接面向 AOI 判定的工业模型，输出必须人工校验 |
| AOI 适合场景 | 冷启动标注、快速生成 mask 初稿、辅助修标 |

SAM 辅助标注 pipeline：

```text
原图
 -> 人工点选或 bbox prompt
 -> SAM 生成 mask
 -> 人工检查和修正
 -> 导出标注
 -> 训练 U-Net / DeepLab / Mask R-CNN
```

性能特点：

- 标注提效明显。
- 直接上线做判定风险较高。
- 对透明、反光、低对比度工业缺陷，需要人工严格审核。

## 4. 异常检测模型

异常检测适合缺陷样本少、正常样本多的场景。常见输出是 image-level anomaly score 和 pixel-level anomaly map。

典型输入输出：

```text
输入：
image: [B, C, H, W]

输出：
anomaly_score: [B]
anomaly_map: [B, 1, H, W]
```

### 4.1 PatchCore

PatchCore 用预训练 CNN 提取 patch 特征，用正常样本构建 memory bank。测试时，如果某个 patch 特征离正常 memory bank 很远，就认为异常。

| 维度 | 说明 |
| --- | --- |
| 输入 | 正常样本训练图；测试图 |
| 输出 | anomaly score、anomaly map |
| 优点 | 冷启动强，定位效果好，工业异常检测经典 baseline |
| 缺点 | memory bank 占用和最近邻检索需要优化 |
| AOI 适合场景 | 缺陷样本少、需要异常热力图、正常状态较稳定 |

PatchCore pipeline：

```text
正常图
 -> 预训练 backbone 提取 patch 特征
 -> coreset 采样
 -> memory bank

测试图
 -> 提取 patch 特征
 -> 最近邻距离
 -> anomaly map
 -> 阈值 + 后处理
 -> OK/NG + 异常区域
```

性能特点：

- 训练成本低，主要是特征提取和建库。
- 推理速度受 memory bank 大小和最近邻检索影响。
- 可用 coreset、FAISS、降维等方式优化。

### 4.2 EfficientAD

EfficientAD 使用轻量 teacher-student 和 autoencoder 思路，同时关注局部异常和全局异常，目标是低延迟异常检测。

| 维度 | 说明 |
| --- | --- |
| 输入 | 正常样本训练图；测试图 |
| 输出 | anomaly score、anomaly map |
| 优点 | 速度快，适合实时场景 |
| 缺点 | 训练、归一化和阈值策略需要仔细验证 |
| AOI 适合场景 | 边缘部署、实时异常检测、冷启动 |

EfficientAD pipeline：

```text
正常图
 -> teacher 提取目标特征
 -> student 学习正常特征
 -> autoencoder 学习全局结构

测试图
 -> teacher/student 差异
 -> autoencoder 差异
 -> anomaly map
 -> 阈值判定
```

性能特点：

- 推理延迟低，是异常检测中较适合产线部署的方案。
- 对正常样本覆盖度敏感。
- 阈值需要结合 hard normal 和少量缺陷样本调优。

### 4.3 PaDiM

PaDiM 对正常样本的 patch 特征建高斯分布，测试时用 Mahalanobis distance 计算异常程度。

| 维度 | 说明 |
| --- | --- |
| 输入 | 正常训练图；测试图 |
| 输出 | anomaly map、anomaly score |
| 优点 | 思路清晰，适合理解正常分布建模 |
| 缺点 | 高维协方差估计和复杂正常变化会带来问题 |
| AOI 适合场景 | 中小规模异常检测、教学和 baseline |

PaDiM pipeline：

```text
正常图
 -> backbone 特征
 -> 每个空间位置估计均值和协方差

测试图
 -> backbone 特征
 -> Mahalanobis distance
 -> anomaly map
 -> 阈值判定
```

性能特点：

- 推理相对直接。
- 需要存储每个位置的分布参数。
- 对配准要求较高，因为它隐含空间位置对应关系。

### 4.4 STFPM

STFPM 使用 teacher-student feature pyramid matching。Teacher 固定，Student 学习正常样本上的 teacher 多尺度特征。测试时，两者差异越大，越可能异常。

| 维度 | 说明 |
| --- | --- |
| 输入 | 正常训练图；测试图 |
| 输出 | 多尺度差异图、anomaly map |
| 优点 | 多尺度异常直观，适合纹理和结构异常 |
| 缺点 | 训练稳定性、特征层选择和阈值需要调试 |
| AOI 适合场景 | 表面纹理异常、结构异常、缺陷样本少 |

STFPM pipeline：

```text
正常图
 -> teacher features
 -> student features
 -> student 学 teacher

测试图
 -> teacher/student feature difference
 -> 多尺度融合
 -> anomaly map
 -> 后处理和判定
```

性能特点：

- 推理速度取决于 teacher 和 student backbone。
- 多尺度融合对小缺陷有帮助。
- 对正常样本多样性和训练收敛敏感。

### 4.5 FastFlow

FastFlow 用 normalizing flow 建模正常特征分布。正常样本特征在 flow 变换后应符合简单分布，异常样本会表现为低 likelihood 或高异常分数。

| 维度 | 说明 |
| --- | --- |
| 输入 | 正常训练图；测试图 |
| 输出 | likelihood、anomaly map、anomaly score |
| 优点 | 分布建模能力强，可输出像素级异常图 |
| 缺点 | 实现和调参复杂，训练稳定性需要关注 |
| AOI 适合场景 | 想进一步提升异常分布建模能力的进阶方案 |

FastFlow pipeline：

```text
正常图
 -> backbone 特征
 -> flow 模型学习正常特征分布

测试图
 -> backbone 特征
 -> flow likelihood
 -> anomaly map
 -> 阈值和后处理
```

性能特点：

- 比 PatchCore 更偏训练式分布建模。
- 推理速度取决于 flow 结构和特征分辨率。
- 需要更多调参经验，不建议作为第一个 AOI baseline。

## 5. 高分辨率小缺陷方案

高分辨率小缺陷不是单一模型问题，而是输入组织方式和后处理问题。

典型输入输出：

```text
输入：
large_image: [H_large, W_large, C]

中间结果：
tiles: [N, tile_h, tile_w, C]

输出：
原图坐标系下的 bbox / mask / anomaly map / OK-NG
```

### 5.1 Sliding Window

Sliding window 用固定窗口在整张大图上滑动，把每个窗口送入模型。

| 维度 | 说明 |
| --- | --- |
| 优点 | 简单通用，不依赖先验 ROI |
| 缺点 | 计算量大，重复区域多 |
| 适合场景 | 早期 baseline、缺陷可能出现在任意位置 |

pipeline：

```text
大图
 -> 固定窗口滑动
 -> 每个窗口推理
 -> 窗口坐标还原
 -> 跨窗口 NMS / mask 合并
 -> 整图判定
```

性能特点：

- stride 越小，召回越高，推理越慢。
- overlap 可以降低边缘漏检。
- 适合先证明问题可解，再优化为 ROI 或 cascade。

### 5.2 Tiling

Tiling 把大图切成固定大小 tile，分别做分类、检测、分割或异常检测。

| 维度 | 说明 |
| --- | --- |
| 优点 | 保留局部细节，适配常规模型输入 |
| 缺点 | 需要处理 overlap、边缘截断、坐标还原和重复检测 |
| 适合场景 | 4K/8K 图、线扫图、小划伤、小异物 |

pipeline：

```text
大图
 -> ROI
 -> tile 切图 + overlap
 -> batch 推理
 -> tile 结果还原原图坐标
 -> NMS / mask union / 连通域合并
 -> 业务判定
```

性能特点：

- tile size 影响上下文和速度。
- overlap 影响边缘缺陷召回。
- batch 推理可以显著提高 GPU 利用率。
- tile 数量通常是延迟瓶颈。

### 5.3 ROI Cascade

ROI cascade 先用轻量规则或模型粗筛可疑区域，再用更强模型精检。

| 维度 | 说明 |
| --- | --- |
| 优点 | 大幅减少无效区域推理，速度快 |
| 缺点 | 第一阶段漏掉的缺陷无法被第二阶段找回 |
| 适合场景 | 缺陷稀疏、背景区域大、ROI 有先验 |

pipeline：

```text
大图
 -> 第一阶段：规则 / 轻量模型找候选
 -> 扩展候选 ROI
 -> 第二阶段：检测 / 分割 / 分类复核
 -> 坐标还原
 -> 业务判定
```

性能特点：

- 第一阶段必须高召回。
- 第二阶段可以更重、更精细。
- 适合节拍紧但图像很大的场景。

### 5.4 多尺度推理

多尺度推理用多个输入尺度或多尺度特征检测不同大小的缺陷。

| 维度 | 说明 |
| --- | --- |
| 优点 | 兼顾大小缺陷 |
| 缺点 | 计算成本高，合并策略复杂 |
| 适合场景 | 同一产品上缺陷尺度差异很大 |

pipeline：

```text
原图 / ROI
 -> 多个尺度 resize 或多尺度 tile
 -> 多尺度推理
 -> 坐标统一到原图
 -> 结果融合
 -> NMS / mask 合并
```

性能特点：

- 精度可能提升，但延迟增加明显。
- 线上通常只保留验证收益明确的尺度。
- 可以先离线使用多尺度评估上限，再设计轻量线上方案。

## 6. 规则 + 模型混合方案

强几何约束 AOI 不应只依赖端到端深度模型。规则和模型各有优势：

```text
规则：可解释、快、适合尺寸位置测量
模型：适合纹理、语义、复杂外观判断
```

典型输入输出：

```text
输入：
image + calibration + ROI config + rule config

输出：
result: OK / NG / REVIEW
measurements: 尺寸、位置、角度、面积
model_scores: 模型复核分数
evidence: bbox / mask / 轮廓 / 测量线
```

### 6.1 模板配准

模板配准用于把当前图对齐到标准坐标系，让后续 ROI 和测量稳定。

| 维度 | 说明 |
| --- | --- |
| 输入 | 当前图、模板图或定位特征 |
| 输出 | 位移、旋转、透视变换矩阵、动态 ROI |
| 优点 | 提高 ROI 稳定性，减少位置偏移误报 |
| 缺点 | 对反光、遮挡、形变和模板变化敏感 |
| 适合场景 | 固定产品、固定工位、需要精确测量 |

pipeline：

```text
当前图
 -> 找定位点 / 模板匹配 / 特征匹配
 -> 估计变换
 -> 图像或 ROI 对齐
 -> 后续规则 / 模型检测
```

性能特点：

- 通常很快，适合实时前处理。
- 配准失败要有兜底逻辑，例如进入复判或报警。

### 6.2 几何测量

几何测量用传统视觉计算尺寸、位置、距离、角度、面积等。

| 维度 | 说明 |
| --- | --- |
| 输入 | 校正后图像、ROI、像素到物理尺寸标定 |
| 输出 | 测量值、是否超公差、证据图 |
| 优点 | 可解释、速度快、结果可追溯 |
| 缺点 | 对光照、边缘质量、阈值和定位稳定性敏感 |
| 适合场景 | 孔位、间距、宽度、边缘缺口、胶水越界 |

pipeline：

```text
ROI
 -> 光照归一化
 -> 阈值 / 边缘 / 轮廓
 -> 拟合线 / 圆 / 矩形
 -> 像素转物理单位
 -> 公差判定
```

性能特点：

- CPU 上也可以很快。
- 对成像一致性要求高。
- 适合和深度模型组合，规则先做候选，模型做复核。

### 6.3 模型复核

模型复核是在规则找到候选后，用分类、检测或分割模型判断候选是否真缺陷。

| 维度 | 说明 |
| --- | --- |
| 输入 | 规则候选 ROI、候选图、测量信息 |
| 输出 | 模型分数、复核类别、最终判定 |
| 优点 | 降低传统规则误报，增强纹理和语义判断 |
| 缺点 | 需要维护规则和模型两套版本 |
| 适合场景 | 规则召回高但误报多的 AOI |

pipeline：

```text
原图
 -> 规则找候选
 -> 裁剪候选 ROI
 -> 分类 / 检测 / 分割模型复核
 -> 融合规则分数和模型分数
 -> OK / NG / REVIEW
```

性能特点：

- 比整图模型推理更快，因为只看候选区域。
- 第一阶段规则必须保证高召回。
- 适合构建可解释且可控的产线系统。

## 7. 性能对比与选型参考

| 方案 | 输出粒度 | 标注成本 | 小缺陷能力 | 延迟 | 部署复杂度 | 典型用途 |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet / EfficientNet / ConvNeXt / ViT 分类 | image-level | 低 | 依赖 ROI/tile | 低到中 | 低 | OK/NG、缺陷类别 |
| YOLO 检测 | bbox-level | 中 | 需高分辨率或 tile | 低 | 中 | 实时缺陷定位 |
| Faster R-CNN 检测 | bbox-level | 中 | 较好 | 中到高 | 中到高 | 精度优先检测 |
| RT-DETR 检测 | bbox-level | 中 | 中到较好 | 中 | 中到高 | 复杂关系检测 |
| U-Net / DeepLab 分割 | pixel-level | 高 | 取决于输入和 loss | 中 | 中 | 面积、边界、形态 |
| Mask R-CNN | instance mask | 高 | 中 | 高 | 高 | 实例级缺陷 mask |
| PatchCore / PaDiM | anomaly map | 低 | 较好 | 中 | 中 | 冷启动异常检测 |
| EfficientAD | anomaly map | 低 | 较好 | 低 | 中 | 实时异常检测 |
| FastFlow | anomaly map | 低 | 较好 | 中 | 中到高 | 进阶异常检测 |
| Tiling / ROI cascade | 原图坐标结果 | 取决于模型 | 强 | 中到高 | 中 | 大图小缺陷 |
| 规则 + 模型混合 | measurement + score | 中 | 强依赖规则 | 低到中 | 中到高 | 强几何约束 AOI |

## 8. 上线时必须拆开的性能指标

不要只报模型推理时间。AOI 线上系统要拆开：

```text
采集耗时
图像解码耗时
质量检查耗时
ROI / 配准耗时
tile 切图耗时
预处理耗时
模型推理耗时
后处理耗时
坐标还原耗时
规则判定耗时
PLC / MES / HMI 通信耗时
总延迟 P50 / P95 / P99
```

性能优化顺序通常是：

```text
先减少无效输入
 -> 再优化 batch 和 tile
 -> 再导出 ONNX / TensorRT / OpenVINO
 -> 再考虑 FP16 / INT8
 -> 最后换更复杂的模型
```

## 9. 快速选型建议

- 只判断整图 OK/NG：先用 ResNet50 或 ConvNeXt-Tiny。
- 边缘设备、算力紧：EfficientNet-B0/B1 或轻量 YOLO。
- 要缺陷位置：先用 YOLO。
- 要面积、长度、边界：先用 U-Net 或 DeepLab。
- 要区分多个缺陷实例：考虑 Mask R-CNN 或检测 + 分割组合。
- 缺陷样本少：先用 PatchCore 或 EfficientAD。
- 大图小缺陷：先设计 tile 和坐标还原，再谈模型。
- 有明确尺寸公差：先做标定、配准和几何测量，再用模型复核。

