# EEG 压力/情绪识别 — 混合 CNN-LSTM 模型

基于三篇核心论文实现的 EEG 信号压力量化与情绪识别框架：

| 论文 | 来源 | 贡献 |
|------|------|------|
| **Chaudhari & Shrivastava (2026)** — *Hybrid CNN–LSTM Model For Continuous Stress Quantification Via Emotion-Valence Mapping On EEG Signals* | Int. J.Adv.Sig.Img.Sci, Vol. 12, No. 2s | 连续压力量化、情绪-价值映射、STEW 100%/Neurocom 97% 准确率 |
| **Zhu, Song & Li (2024)** — *EEG Emotion Recognition Based on CNN+LSTM* | IEEE CCSSTA 2024 | 差分熵(DE)特征、电极-矩形映射、Valence 95.82%/Arousal 95.96% |
| **Choudhary et al. (2025)** — *Hybrid CNN-LSTM Model for EEG-Based Emotion Recognition: A Comparative Analysis Using DEAP and SEED Datasets* | IEEE IC3IT 2025 | GAN数据增强、PCA+MI特征选择、DEAP 93.4%/SEED 91.2% |

---

## 项目概述

本项目实现了一个混合 **CNN-LSTM** 深度学习架构，用于基于 EEG 信号的心理压力识别和情绪分类。主要特点：

- **双任务支持**：回归（连续压力值 0~1）+ 分类（情绪标签）
- **多数据集兼容**：DEAP (32通道)、SEED (62通道)、STEW (14通道)、Neurocom
- **差分熵(DE)特征**：论文[2]方法，四个频段提取 (Theta/Alpha/Beta/Gamma)
- **电极-矩形映射**：论文[2]证明可提升空间特征判别性
- **数据增强**：高斯噪声、通道Dropout、时间掩码
- **频带能量分析**：Delta/Theta/Alpha/Beta/Gamma 五频段
- **集成模型**：时域+频域融合预测（论文[1]）
- **YAML配置驱动**：所有超参数通过 `config.yaml` 管理
- **完整评估指标**：回归(MSE/MAE/R²/Corr) + 分类(Accuracy/Precision/Recall/F1)
- **实时推理引擎**：支持W8脑电放大器实时采集→预处理→模型推理→压力指数
- **实时压力检测GUI**：PyQt5图形界面，EEG信号波形+压力等级指示+趋势图
- **实时FFT频谱显示**：PyQt5图形界面，EEG信号波形+FFT频谱分析（`test1.py`）

---

## 最新进展

### 实时压力检测GUI（2026年6月）

新增 `inference.py` 实时压力检测图形界面，基于 PyQt5 + Matplotlib：

- **EEG信号波形**：实时显示滤波后脑电信号（可选通道FCz/Pz/POz/O1/Oz/O2）
- **压力等级指示**：大号彩色文字显示放松/中等压力/高压，附带置信度进度条
- **压力趋势图**：记录最近80次推理结果，展示压力等级随时间变化
- **按钮控制**：▶ 开始 / ■ 停止采集 + 6通道选择

数据流：
```
W8 (500Hz, 8ch)
  ├→ 显示管线: 重参考 → 带通2-150Hz+陷波 → 信号波形
  └→ 推理引擎: 重参考 → 降采样125Hz → 带通0.5-45Hz → 通道映射
              → z-score归一化 → 2.5s滑窗 → 模型推理 → 3类压力(放松/中等/高压)
```

### 实时FFT频谱显示

新增 `test1.py` 实时EEG信号+FFT频谱显示程序，同硬件平台，用于信号质量检查和频段分析。

### 问题修复与优化（2026年5月）

项目针对 STEW 数据集训练中发现的 7 个问题进行了系统性修复：

| 问题 | 严重程度 | 解决方案 |
|------|---------|---------|
| 数据泄露（滑窗重叠导致相邻段信息共享） | P0 | overlap=0，不重叠滑窗，消除数据泄露 |
| 类别不均衡 | P1 | WeightedRandomSampler + 权重上限(5倍中位数) |
| 过拟合 | P2 | dropout=0.4, weight_decay=0.03, label_smoothing=0.2 |
| 数据增强不足 | P3 | 增强噪声(std=0.05)、通道dropout(0.1)、时间掩码(size=40) |
| 训练不稳定 | P4 | ReduceLROnPlateau + 10轮warmup |
| 模型过大 | P5 | CNN通道[32,64,128]，LSTM隐层64 |
| 标签缺失(0-3类) | P6 | 不重叠滑窗后，所有4-9类均有样本分配到测试集 |

**最终结果（10分类）：测试准确率 64.49%**（加权平均F1 0.66）

### 3分类优化（后续实验）

将10类标签映射为低/中/高三档压力后：

| 指标 | 值 |
|------|----|
| 学习率 | 0.0005（10分类的0.001→0.0005） |
| 权重衰减 | 0.05（增强正则化） |
| 标签平滑 | 0.05（3类不需要0.2的强平滑） |
| 早停耐心 | 20轮 |

---

## 项目结构

```
eeg_stress_detection/
├── config.yaml              # 统一配置文件（模型/训练/增强/数据划分）
├── model.py                 # 模型定义（CNN-LSTM, 频域, 集成, GAN）
├── trainer.py               # 训练器, 数据集, 数据增强, 交叉验证
├── dataset_transform.py     # 数据集加载与预处理工具
├── train_stew.py            # STEW数据集 10分类训练脚本
├── train_6ch_model.py       # 6通道(适配W8) 3分类训练脚本
├── inference.py             # 实时压力检测GUI（W8脑电放大器→压力等级）
├── test1.py                 # 实时EEG信号+FFT频谱显示（同硬件平台）
├── example.py               # 使用示例和演示（原demo脚本）
├── problem.txt              # 问题记录
├── requirements.txt         # 依赖包列表
├── README.md                # 项目说明
│
├── stew_logs/               # STEW 10分类训练日志（TensorBoard + 图表）
│   ├── training.log
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── events.out.tfevents.*
│
├── stew_models/             # STEW 10分类模型检查点
│   ├── best_model.pth
│   └── latest_checkpoint.pth
│
├── stew_logs_6ch/           # 6通道 3分类训练日志
└── stew_models_6ch/         # 6通道 3分类模型检查点
```

---

## 模型架构

### EEGStressCNNLSTM (主模型)

基于三篇论文的混合架构：

```
输入: [batch, n_channels, n_timepoints]  (默认 14×320)
  │
  ├─ [可选] 2D电极-矩形映射 (论文[2] Fig.3)
  │    └─ 2D-CNN 空间特征提取（DEAP 32ch / SEED 62ch）
  │
  ├─ 1D-CNN 空间/频率特征提取 (论文[2] 第三节C)
  │    ├─ Conv1D(k=7) → BN → GELU → Dropout(0.4)
  │    ├─ Conv1D(k=5) → BN → GELU → Dropout(0.4)
  │    ├─ Conv1D(k=5) → BN → GELU → Dropout(0.4)
  │    └─ AdaptiveMaxPool1d (降采样至 n_timepoints//4)
  │
  ├─ 注意力机制
  │    ├─ 通道注意力 (ChannelAttention)
  │    └─ 空间注意力 (SpatialAttention)
  │
  ├─ Bi-LSTM 时序建模 (论文[2] 第三节D)
  │    └─ 2层 LSTM, hidden=64, bidirectional
  │
  ├─ 多头自注意力 (MultiheadAttention, heads=4)
  │
  └─ 输出头
       ├─ 回归: Linear → Sigmoid → [0,1] 压力值 (论文[1])
       └─ 分类: Linear → Logits → Softmax (论文[2])
```

### FrequencyDomainEEGModel
频域特征专用模型，接收各频段特征向量。

### EnsembleEEGModel
时域+频域融合模型（论文[1]），通过融合层合并两个子模型的预测。

### GAN数据增强 (论文[3] 第三节)
- **生成器**: 全连接网络，从噪声生成合成EEG信号
- **判别器**: 卷积网络，区分真实/合成信号

---

## 快速开始

### 环境设置

```bash
# 创建虚拟环境 (推荐)
python -m venv eeg_env
source eeg_env/bin/activate  # Linux/Mac
# 或
eeg_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### STEW 10分类训练

```bash
# 使用默认配置
python train_stew.py

# 使用自定义配置
python train_stew.py --config my_config.yaml
```

### 6通道(W8适配) 3分类训练

```bash
python train_6ch_model.py
```

### 实时推理（无GUI模式）

```python
from inference import StressInferenceEngine

# 加载训练好的6通道模型
engine = StressInferenceEngine(model_path="./stew_models_6ch/best_model.pth")

# 逐帧输入W8原始数据 (8通道, 500Hz)
for frame_8ch in w8_device_stream():
    result = engine.feed_frame(frame_8ch)  # 自动重参考→降采样→滤波→推理
    if result:
        level = result["stress_level"]  # 0=放松, 1=中等压力, 2=高压
        probs = result["probabilities"]  # [p_low, p_med, p_high]
        names = ["放松", "中等压力", "高压"]
        print(f"压力等级: {names[level]} (置信度: {probs[level]*100:.0f}%)")
```

### 离线推理测试

```python
from inference import StressInferenceEngine
import numpy as np

engine = StressInferenceEngine(model_path=None)  # 随机权重

# 模拟W8数据 (8通道, 500Hz, 5秒)
t = np.arange(0, 5, 1/500)
raw_data = np.random.randn(len(t), 8) * 2.0

# 逐帧输入
for frame in raw_data:
    result = engine.feed_frame(frame)
    if result:
        print(f"压力等级: {result['stress_level']} (0=放松/1=中等/2=高压)")

### 数据加载与预处理

```python
from dataset_transform import load_stew_format, segment_eeg, preprocess_eeg

# 加载STEW数据
eeg, labels = load_stew_format("./data")  # (45, 14, 19200)

# 预处理：z-score归一化 + 0.5-45Hz带通滤波
eeg = preprocess_eeg(eeg, normalize="zscore", bandpass_low=0.5, bandpass_high=45, fs=128)

# 滑窗分段：2.5秒窗口，不重叠
segs, seg_labels = segment_eeg(eeg, labels, window_sec=2.5, fs=128, overlap=0.0)
# 输出: (2700, 14, 320)
```

---

## 配置系统

所有训练和模型参数通过 `config.yaml` 管理，包含详细注释说明每个参数的影响。

### 关键配置项

| 模块 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 任务 | `num_classes` | 3 | 分类数（10类/3类/2类，默认3类适配W8） |
| 预处理 | `window_sec` | 2.5 | 滑窗长度(秒) |
| | `overlap` | 0.0 | 滑窗重叠率（0=不重叠） |
| | `normalize` | zscore | 归一化方式 |
| | `bandpass_low` | 0.5 | 高通截止(Hz) |
| | `bandpass_high` | 45 | 低通截止(Hz) |
| 模型 | `cnn_channels` | [32,64,128] | CNN各层输出通道 |
| | `lstm_hidden` | 64 | LSTM隐层大小 |
| | `lstm_layers` | 2 | LSTM层数 |
| | `dropout_rate` | 0.4 | Dropout比率 |
| | `activation` | gelu | 激活函数 |
| 训练 | `epochs` | 50 | 最大训练轮数 |
| | `batch_size` | 64 | 批次大小 |
| | `learning_rate` | 0.001 | 初始学习率 |
| | `weight_decay` | 0.03 | 权重衰减 |
| | `lr_scheduler` | plateau | 学习率调度策略 |
| | `label_smoothing` | 0.2 | 标签平滑系数 |
| 增强 | `noise_std` | 0.05 | 高斯噪声标准差 |
| | `channel_dropout_prob` | 0.1 | 通道随机丢弃概率 |
| | `time_mask_prob` | 0.1 | 时间掩码概率 |
| | `time_mask_size` | 40 | 时间掩码长度 |

---

## 实时压力检测GUI (`inference.py`)

专为 **W8 脑电放大器** 设计的实时压力检测图形界面。

### 启动

```bash
python inference.py
```

### GUI布局

```
┌───────────────────────────────────────────────┐
│           EEG 信号波形（实时滤波后）           │
│           (可选通道 FCz/Pz/POz/O1/Oz/O2)      │
├───────────────────────────────────────────────┤
│           压力等级: 放松 / 中等压力 / 高压     │
│           ████████████░░░ 置信度: 87%          │
├───────────────────────────────────────────────┤
│           压力趋势图（最近~40秒变化）          │
│  2 ┤        ▄▄                              │
│  1 ┤▄▄▄▄▄▄▄▄  ▀▀▀▀▄▄▄▄▄▄▄▄▄▄               │
│  0 ┤──────────────────────────▀▀▀▀▀▀▀▀▀▀▀▀  │
├───────────────────────────────────────────────┤
│  [▶ 开始] [FCz] [Pz] [POz] [O1] [Oz] [O2]  │
└───────────────────────────────────────────────┘
```

### 数据流

```
W8 (500Hz, 8通道)
  → 重参考: (TP9 + TP10) / 2
  → 6信号通道: [FCz, Pz, POz, O1, Oz, O2]
  
  ┌─ 显示管线 ──────────────────────────────┐
  │  带通2-150Hz + 陷波48-52Hz + 陷波98-102Hz │
  │  → 实时EEG信号波形                       │
  └──────────────────────────────────────────┘
  
  ┌─ 推理管线 ──────────────────────────────┐
  │  降采样: 500Hz → 125Hz                  │
  │  → 带通滤波: 0.5-45Hz (Butterworth 4阶)  │
  │  → 通道映射: W8通道 → 模型输入顺序       │
  │  → z-score归一化                          │
  │  → 2.5秒滑窗 (312点 @125Hz)              │
  │  → CNN-LSTM推理 → 3类压力(放松/中等/高压)  │
  │  → 3帧平滑输出                            │
  │  → 趋势图更新                             │
  └──────────────────────────────────────────┘
```

### 关键特性

- **实时EEG波形**：500Hz采样，2-150Hz带通+陷波滤波，4秒滑动窗口
- **压力等级指示**：彩色大号文字（绿=放松/橙=中等/红=高压）+ 置信度进度条
- **压力趋势图**：记录最近80次推理（~40秒），步进图显示压力变化趋势
- **通道选择**：6个信号通道任意切换查看
- **推理节流**：每0.5秒推理一次，3帧平滑防止突变
- **模型自动加载**：从 `stew_models_6ch/best_model.pth` 加载3分类模型

---

## 数据集支持

### STEW (主数据集)

| 属性 | 值 |
|------|-----|
| 通道数 | 14 EEG (Emotiv EPOC) |
| 采样率 | 128 Hz |
| 被试 | 45人 × 150秒 Stroop任务 |
| 标签 | 0-9 十级压力评分 |
| 实际有效类 | 4-9（6类，0-3类无被试数据） |
| 预滑动窗后样本量 | 2700段 (2.5s, 不重叠) |

标签映射（3分类方案）：
```
原始0-9 → 低压力(0-3→0) / 中压力(4-6→1) / 高压力(7-9→2)
```

6通道选择（适配W8脑电放大器）：
```
FC5(3), P7(5), O1(6), O2(7), P8(8), FC6(10)
```

### DEAP (论文[2][3])

| 属性 | 值 |
|------|-----|
| 通道数 | 32 EEG |
| 采样率 | 128 Hz |
| 实验 | 32人 × 40视频 (60秒) |
| 标签 | Valence, Arousal, Dominance, Liking (1~9) |

### SEED (论文[3])

| 属性 | 值 |
|------|-----|
| 通道数 | 62 EEG |
| 采样率 | 200 Hz (原始1000Hz降采样) |
| 标签 | Positive / Neutral / Negative |

### 加载示例

```python
from dataset_transform import load_stew_format, load_deap_format, load_seed_format

# STEW
data, labels = load_stew_format("./data")  # 返回 .mat 文件

# DEAP
data, labels = load_deap_format("./data/deap/s01.mat")  # 返回 .dat 文件

# SEED
data, labels = load_seed_format("./data/seed/1_20131027.mat")  # 返回 .mat/.h5
```

---

## 特征提取

### 差分熵(DE) (论文[2] 公式1-3)

DE(x) = ½ · log(2πeσ²)

对每个频段内的EEG信号计算，适用于四频段：
- **Theta**: 4~8 Hz
- **Alpha**: 8~14 Hz
- **Beta**: 14~31 Hz
- **Gamma**: 31~45 Hz

```python
from model import compute_differential_entropy, extract_band_de_features

de = compute_differential_entropy(eeg_signal)             # [B, C]
band_de = extract_band_de_features(eeg_signal, fs=128)    # [B, C, 4]
```

### 频带能量

```python
from dataset_transform import extract_freq_band_power

band_power = extract_freq_band_power(eeg_data, fs=128)
# 返回: [B, C, 5] — Delta, Theta, Alpha, Beta, Gamma
```

---

## 评估指标

### 分类任务（STEW 10分类）

| 类别 | 原始标签 | 精确率 | 召回率 | F1 | 样本数 |
|------|---------|--------|--------|-----|-------|
| 4 | 低压力 | 0.32 | 0.87 | 0.47 | 54 |
| 5 | 中压力 | 0.83 | 0.59 | 0.69 | 180 |
| 6 | 中压力 | 0.56 | 0.77 | 0.65 | 126 |
| 7 | 高压力 | 0.75 | 0.57 | 0.65 | 198 |
| 8 | 高压力 | 0.77 | 0.60 | 0.67 | 199 |
| 9 | 高压力 | 0.71 | 0.78 | 0.74 | 54 |

**加权平均 F1: 0.66  |  准确率: 64.49%**（overlap=0, 无数据泄露）

### 回归任务

| 指标 | 含义 |
|------|------|
| MSE | 均方误差 ↓ |
| MAE | 平均绝对误差 ↓ |
| R² | 决定系数 ↑ |
| Correlation | 相关系数 ↑ |

### 论文基准

| 模型 | DEAP Valence | DEAP Arousal | STEW | Neurocom |
|------|-------------|--------------|------|----------|
| SVM | 84.0% | — | — | — |
| CNN | 90.1% | 88.5% | — | — |
| LSTM | 88.4% | 86.2% | — | — |
| PCRNN | 90.26% | 90.98% | — | — |
| 论文[2]模型 | **95.82%** | **95.96%** | — | — |
| 论文[1]模型 | — | — | **100%** | **97%** |

---

## 训练日志与可视化

每次训练自动生成：

- **TensorBoard日志**：`stew_logs/events.out.tfevents.*`，可用 `tensorboard --logdir stew_logs` 查看
- **训练曲线**：`stew_logs/loss_curve.png`（损失和准确率变化）
- **混淆矩阵**：`stew_logs/confusion_matrix.png`
- **分类报告**：`stew_logs/classification_report.txt`
- **训练日志**：`stew_logs/training.log`（每轮详细指标）

### Visdom 实时可视化

```bash
# 启动 Visdom 服务
python -m visdom.server -port 707

# 训练时自动连接 (config.yaml 中 visdom.enabled=true)
```

---

## 引用

如果您在研究中使用了此代码，请引用相关论文：

```bibtex
# 论文[1] — 连续压力量化
@article{chaudhari2026hybrid,
  title={Hybrid CNN--LSTM Model For Continuous Stress Quantification Via
         Emotion-Valence Mapping On EEG Signals},
  author={Chaudhari, Amol and Shrivastava, Hemang},
  journal={Int. J.Adv.Sig.Img.Sci},
  volume={12},
  number={2s},
  year={2026}
}

# 论文[2] — EEG情绪识别
@inproceedings{zhu2024eeg,
  title={EEG Emotion Recognition Based on CNN+LSTM},
  author={Zhu, Xuanpeng and Song, Yu and Li, Dong},
  booktitle={2024 IEEE 25th China Conference on System Simulation Technology
             and its Application (CCSSTA)},
  year={2024},
  doi={10.1109/CCSSTA62096.2024.10691696}
}

# 论文[3] — DEAP与SEED比较分析
@inproceedings{choudhary2025hybrid,
  title={Hybrid CNN-LSTM Model for EEG-Based Emotion Recognition:
         A Comparative Analysis Using DEAP and SEED Datasets},
  author={Choudhary, Alekh and Das, Papri and Sharma, Vikas and
          Vashishth, Tarun Kumar and Vidyant, Sanjukta and Kumar, Sunil},
  booktitle={2025 International Conference on Communication, Computer,
             and Information Technology (IC3IT)},
  year={2025},
  doi={10.1109/IC3IT66137.2025.11341346}
}
```

---

## 许可证

MIT License
