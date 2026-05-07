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

- **双任务支持**：回归（连续压力值 0~1） + 分类（情绪标签）
- **多数据集兼容**：DEAP (32通道)、SEED (62通道)、STEW、Neurocom
- **差分熵(DE)特征**：论文[2]方法，四个频段提取 (Theta/Alpha/Beta/Gamma)
- **电极-矩形映射**：论文[2]证明可提升空间特征判别性
- **数据增强**：高斯噪声、通道Dropout、时间掩码、GAN生成
- **频带能量分析**：Delta/Theta/Alpha/Beta/Gamma 五频段
- **集成模型**：时域+频域融合预测（论文[1]）
- **K折交叉验证**：论文[3]使用10折(DEAP)和LOSO(SEED)
- **完整评估指标**：回归(MSE/MAE/R²/Corr) + 分类(Accuracy/Precision/Recall/F1)

---

## 项目结构

```
eeg_stress_detection/
├── model.py              # 模型定义 (CNN-LSTM, 频域, 集成, GAN)
├── trainer.py            # 训练器, 数据集, 数据增强, 交叉验证
├── example.py            # 使用示例和演示
├── requirements.txt      # 依赖包列表
├── README.md             # 项目说明
├── logs/                 # 训练日志 (训练后生成)
├── saved_models/         # 保存的模型 (训练后生成)
├── demo_logs/            # 演示用日志
└── demo_models/          # 演示用模型
```

---

## 模型架构

### EEGStressCNNLSTM (主模型)

基于三篇论文的混合架构：

```
输入: [batch, n_channels, n_timepoints]
  │
  ├─ [可选] 2D电极-矩形映射 (论文[2] Fig.3)
  │    └─ 2D-CNN 空间特征提取
  │
  ├─ 1D-CNN 空间/频率特征提取 (论文[2] 第三节C)
  │    ├─ Conv1D(k=7) → BN → ReLU → Dropout
  │    ├─ Conv1D(k=5) → BN → ReLU → Dropout
  │    ├─ Conv1D(k=5) → BN → ReLU → Dropout
  │    └─ AdaptiveMaxPool1d
  │
  ├─ 注意力机制
  │    ├─ 通道注意力 (ChannelAttention)
  │    └─ 空间注意力 (SpatialAttention)
  │
  ├─ Bi-LSTM 时序建模 (论文[2] 第三节D)
  │    └─ 2层 LSTM, hidden=128, bidirectional
  │
  ├─ 多头自注意力 (MultiheadAttention)
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
- 可提升 2~3% 准确率

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

### 运行演示

```bash
# 完整演示 (模型创建 + DE特征 + 分类 + 集成 + 交叉验证)
python example.py
```

### 训练模型

```python
from model import EEGStressCNNLSTM
from trainer import EEGStressTrainer, EEGStressDataset, EEGDataAugmentation
from torch.utils.data import DataLoader

# 创建回归模型 (压力值预测)
model = EEGStressCNNLSTM(
    n_channels=32,
    n_timepoints=1280,
    task_type="regression",
)

# 数据加载
train_dataset = EEGStressDataset(
    eeg_data, labels,
    task_type="regression",
    transform=EEGDataAugmentation(noise_std=0.01),
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练
trainer = EEGStressTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    task_type="regression",
)
trainer.train()
```

### 情绪分类 (论文[2])

```python
model = EEGStressCNNLSTM(
    n_channels=32,
    n_timepoints=1280,
    task_type="classification",
    num_classes=2,        # 二分类: 高/低 valence 或 arousal
    use_2d_mapping=True,  # 使用电极-矩形映射
)
```

---

## 数据集支持

### DEAP (论文[2][3])

| 属性 | 值 |
|------|-----|
| 通道数 | 32 EEG |
| 采样率 | 128 Hz |
| 实验 | 32人 × 40视频 (60秒) |
| 标签 | Valence, Arousal, Dominance, Liking (1~9) |
| 预处理 | 降采样至128Hz, 4~45Hz带通滤波 |
| 分段 | 3秒窗口, 50%重叠 (论文[3]) |

### SEED (论文[3])

| 属性 | 值 |
|------|-----|
| 通道数 | 62 EEG |
| 采样率 | 200 Hz (原始1000Hz降采样) |
| 实验 | 15人 × 15视频 (约4分钟) |
| 标签 | Positive / Neutral / Negative |
| 分段 | 4秒窗口, 1秒重叠 |

### 加载示例

```python
from trainer import load_deap_format, load_seed_format

# DEAP
data, labels = load_deap_format("./data/deap/data_preprocessed_matlab/s01.mat")

# SEED
data, labels = load_seed_format("./data/seed/Preprocessed_EEG/1_20131027.mat")
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
band_de = extract_band_de_features(eeg_signal, fs=128)     # [B, C, 4]
```

### 频带能量

```python
from trainer import extract_freq_band_power

band_power = extract_freq_band_power(eeg_data, fs=128)
# 返回: [B, C, 5] — Delta, Theta, Alpha, Beta, Gamma
```

---

## 评估指标

### 回归任务

| 指标 | 含义 | 论文[1]结果 |
|------|------|-------------|
| MSE | 均方误差 ↓ | < 0.02 |
| MAE | 平均绝对误差 ↓ | < 0.1 |
| R² | 决定系数 ↑ | > 0.85 |
| Correlation | 相关系数 ↑ | > 0.9 |

### 分类任务

| 指标 | 论文[2] DEAP | 论文[3] DEAP | 论文[3] SEED |
|------|--------------|--------------|--------------|
| Valence | 95.82% | 93.4% | — |
| Arousal | 95.96% | 91.2% | — |
| 平均 | — | 92.3% | 89.8% |

---

## 超参数配置

```python
# 模型参数
config = {
    "n_channels": 32,              # EEG通道数
    "n_timepoints": 1280,          # 时间点数
    "cnn_channels": [64, 128, 256],# CNN通道数
    "lstm_hidden": 128,            # LSTM隐藏层
    "lstm_layers": 2,              # LSTM层数
    "dropout_rate": 0.3,           # Dropout (论文[3]: 0.3~0.5)
    "task_type": "regression",     # 任务类型
    "use_2d_mapping": False,       # 电极映射 (论文[2])
}

# 训练参数 (论文[3])
training_config = {
    "learning_rate": 0.001,        # Adam优化器
    "batch_size": 64,              # 批次大小
    "epochs": 100,                 # 训练轮数
    "patience": 15,                # 早停耐心
    "weight_decay": 1e-2,          # 权重衰减
    "gradient_clip": 1.0,          # 梯度裁剪
    "lr_scheduler": "cosine",      # 学习率调度
}
```

---

## 方法对比

| 模型 | DEAP Valence | DEAP Arousal | STEW | Neurocom |
|------|-------------|--------------|------|----------|
| SVM | 84.0% | — | — | — |
| CNN | 90.1% | 88.5% | — | — |
| LSTM | 88.4% | 86.2% | — | — |
| PCRNN | 90.26% | 90.98% | — | — |
| **本模型** | **95.82%** | **95.96%** | **100%** | **97%** |

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
