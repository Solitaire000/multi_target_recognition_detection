# GSGProbeNet：GSG 微波探针多任务视觉检测网络

> 面向微波探针台精密测试场景的探针关键点定位、校准片识别分类、针痕检测与分组一体化视觉检测系统。

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [技术方案详解](#技术方案详解)
- [训练策略](#训练策略)
- [实验结果](#实验结果)
- [目录结构](#目录结构)
- [技术栈](#技术栈)

---

## 项目简介

本项目构建了一套基于深度学习的多任务视觉检测网络 **GSGProbeNet**，在单张图像上同步完成三个任务：

1. **探针检测 + 关键点定位**：定位探针旋转框及针尖等 16 个关键结构点；
2. **校准片检测 + 分类**：定位校准片并识别其规格/状态；
3. **针痕检测 + 分组**：检测探针接触后留下的针痕点，并将同一次下压产生的 G-S-G 三点正确归为一组。

## 系统架构

```
输入图像
   │
   ▼
HRNetBackbone（多分辨率并行分支）
   │
   ▼
FPN（256通道，输出 P2/P3/P4/P5 四层特征）
   │
   ├── probe_det   (RotatedFCOSHead，多尺度)
   │        │
   │        ▼ 旋转框 RoI
   │   ┌─────────────────┬──────────────────┬────────────────────┐
   │   │ RoIKeypointHead │  ProbeMaskHead    │  ProbeStateHead     │
   │   │ 16关键点+不确定度 │  实例分割          │ 遮挡率/失焦度/可见比 │
   │   └─────────────────┴──────────────────┴────────────────────┘
   │
   ├── calib_det   (RotatedFCOSHead，多尺度，校准片检测+分类)
   │
   └── scrub_head  (ScrubHeatmapHead，单尺度 P2，针痕检测+关联嵌入分组+角色分类)
```

**设计取舍**：HRNet 相比 ResNet+FPN 计算量更大，但对小目标定位更有优势，适合研发验证阶段追求精度；针痕检测的产线部署版本改用更轻量的 YOLOv8 + OpenCV DNN，兼顾工控机的跨平台部署约束与检测速度。

## 技术方案详解

### 1. 旋转框检测头（RotatedFCOSHead）

Anchor-free 单阶段检测头，在 FCOS 基础上扩展旋转角度回归：

- 分类头偏置先验初始化：

$$
b_{init}=-\log\left(\frac{1-\pi_0}{\pi_0}\right)\approx-4.6\quad(\pi_0=0.01)
$$

- 角度回归用 `tanh` 限幅避免周期性跳变：

$$
\hat\theta=\tanh(\hat d_\theta)\times\frac{\pi}{4}
$$

- 解码融合分类置信度与中心度：

$$
score=\sqrt{\mathrm{sigmoid}(cls)\times \mathrm{sigmoid}(ctr)}
$$

**检测头损失：**

$$
FL(p_t)=-\alpha_y(1-p_t)^{\gamma}\big[y\log p+(1-y)\log(1-p)\big]
$$

$$
L_{bbox}=\mathrm{SmoothL1}(\Delta\hat x,\Delta\hat y)+\mathrm{SmoothL1}(\log\hat w,\log\hat h)
$$

$$
L_{angle}=\mathrm{mean}\big(1-\cos(\hat\theta-\theta)\big),\qquad L_{ctr}=\mathrm{BCEWithLogits}(\hat c,t)
$$

$$
L_{det}=\frac{\sum_{layer}\Big[L_{cls}^{(sum)}+\sum_{pos}(L_{bbox}+L_{ctr}+L_{angle})\Big]}{\max(N_{pos},1)}
$$

### 2. 关键点检测头（RoIKeypointHead）

- **soft-argmax** 解码坐标（可导）：

$$
\mu_{xy}=\mathrm{soft\text{-}argmax}(heatmap;\ \tau=100)
$$

- 每个关键点额外预测**各向同性高斯不确定度**：

$$
L_{nll}=\log(\sigma^2)+\frac{\lVert \mu_{xy}^{px}-g_{xy}^{px}\rVert^2}{\sigma^2+\epsilon}
$$

- 前景加权热图损失：

$$
w_{fg}=g_{heatmap}\times4.0+1.0,\qquad
L_{heatmap}=\frac{\sum\big[(p_{heatmap}-g_{heatmap})^2\cdot w_{fg}\cdot m_{vis}\big]}{\sum m_{vis}}
$$

- 关键点总损失：

$$
L_{keypoint}^{final}=(L_{heatmap}+0.2\,L_{nll})+0.3\,L_{vis}
$$

### 3. 多任务不确定性自适应损失加权

$$
L_{total}=\sum_{i=1}^{5}\left[\frac{1}{2}\exp(-2\log\sigma_i)\cdot L_i+\log\sigma_i\right]
$$

其中 $i\in\{probe\_det,\ keypoint,\ calib\_det,\ probe\_state,\ scrub\}$，$\sigma_i$ 为每个任务一个可学习标量参数。损失越大、越不稳定的任务会被自动降权（同方差不确定性思路，参见 Kendall & Gal, 2018）。训练前 5 个 epoch 关闭该机制（sigma-warmup），先用固定权重稳定训练。

### 4. 针痕检测与关联嵌入分组

针痕检测的难点不只是"检测出点"，更在于**分组**——一次探针下压会在测试面上留下 G-S-G 三个点，需正确判断三点归属。

**关联嵌入损失：**

$$
\bar e_g=\frac{1}{|g|}\sum_{i\in g}e_i,\qquad
L_{pull}=\mathrm{mean}_i\big(\lVert e_i-\bar e_{g(i)}\rVert^2\big)
$$

$$
L_{push}=\mathrm{mean}_{g\neq g'}\Big(\exp\big(-0.5\lVert\bar e_g-\bar e_{g'}\rVert^2\big)\Big)
$$

$$
L_{scrub\_assoc}=L_{pull}+L_{push}+0.3\,L_{role}
$$

推理阶段仅用两个**不含可学习参数的纯几何判据**做最终复核（共线性、间距相等性），保证几何硬约束与数据驱动的分组预测相互独立验证。

**磨损评分**：对分组后的针痕做 Kasa 圆拟合评估圆度、Procrustes 形状残差分析，可选马氏距离异常检测（未标定参考样本时自动退化为纯几何评分，避免给出不可信的结论）。

### 5. 实例分割 Mask 头

$$
L_{mask}=\mathrm{BCEWithLogits}(\hat m,m)+\left(1-\frac{2\sum(pg)+s}{\sum p+\sum g+s}\right)
$$

## 训练策略

### 数据标签生成：合成扰动自监督标注

探针状态头（遮挡率/失焦度/可见比例）的标签几乎无法人工标注，采用程序化合成扰动生成精确已知的监督信号：

| 标签 | 生成方式 | 精确性保证 |
|---|---|---|
| 可见比例 | 旋转框角点与画布相交多边形面积 / 自身面积 | 纯几何计算，精确无误差 |
| 遮挡率 | 仅在真实实例 mask 范围内随机挖 1~3 个矩形洞 | 挖洞面积/mask面积精确已知，避免全图随机挖洞导致的实例归属混淆 |
| 失焦度 | 仅在实例邻域内做已知 $\sigma$ 的局部高斯模糊 | $\sigma/\sigma_{max}$ 精确已知，与全图噪声增强分工明确 |

### 两阶段迁移学习

```
Phase 1（伪标签预训练）  20 epochs, lr=1e-3, AdamW + 余弦退火
Phase 2（人工标注微调）  10 epochs, 分层学习率
  backbone : lr × 0.1（冻结前2层）
  FPN      : lr × 0.5
  任务头    : lr × 1.0
sigma_warmup_epochs = 5
```

### 训练稳定性工程

- 混合精度训练（`autocast` + `GradScaler`）；
- 梯度裁剪：**必须**先 `scaler.unscale_(optimizer)` 反缩放梯度，再做 `clip_grad_norm_`；
- 断点续训：完整保存优化器/调度器/epoch状态。

## 实验结果

| 指标 | 数值 |
|---|---|
| 探针关键点平均定位误差 | ≤ 5 px |
| 校准片分类准确率 | 99% |
| 针痕检测 Precision / Recall / mAP@0.5 | 97.0% / 97.2% / 98.3% |
| 探针实例分离 Precision（分割+聚类方案，已废弃） | 0.177（细长目标假阳性严重） |
| 探针实例分离 F1（切换统一 RotatedFCOS 架构后） | 0.973 |
| 校准片实例分离 F1 | 0.973 |

**关键工程教训**：探针分支最初采用"语义分割 + 偏移向量投票聚类"方案，验证发现细长/密集排布目标的偏移向量投票极易分裂到多个错误候选中心，Precision 仅 0.177；团队将探针分支统一切换为已在校准片分支验证有效的 RotatedFCOSHead 架构后，指标提升至 F1=0.973，同时训练/推理流程更统一、维护成本更低。

## 目录结构

```
.
├── model.py                       # GSGProbeNet 网络定义（backbone/FPN/各检测头）
├── train.py                       # 两阶段训练脚本，含损失函数与多任务加权
├── data_pipeline.py                # 数据增强与合成扰动标签生成
├── metrics.py                     # 评估指标（mAP/F1/PCK等）
├── predict.py                     # 推理脚本
├── scrub_geometry.py               # 针痕分组的纯几何判据
├── wear_scoring.py                 # 磨损评分（圆拟合/Procrustes/马氏距离）
├── cv_baseline.py / scrub_cv_baseline.py  # 冷启动伪标签的传统CV基线方案
├── merge_scrub_pseudo_labels.py    # 伪标签合并工具
├── export_and_cpp_interface.py     # ONNX导出与C++部署接口
└── dataset_config.json             # 数据集配置
```

## 技术栈

`Python` · `PyTorch` · `HRNet` · `FCOS` · `OpenCV` · `ONNX` · `C++` (部署) · `Qt` (上位机界面)

> 本仓库为课题组内相关研究方向的技术记录与代码整理，部分模块为团队协作成果，具体分工请参考项目说明文档。
