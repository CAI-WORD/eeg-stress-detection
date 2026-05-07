#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG压力/情绪识别模型 — 混合CNN-LSTM架构

基于以下论文实现:
  [1] Chaudhari & Shrivastava (2026). Hybrid CNN–LSTM Model For Continuous
      Stress Quantification Via Emotion-Valence Mapping On EEG Signals.
      Int. J.Adv.Sig.Img.Sci, Vol. 12, No. 2s.
  [2] Zhu, Song & Li (2024). EEG Emotion Recognition Based on CNN+LSTM.
      IEEE CCSSTA 2024. DOI: 10.1109/CCSSTA62096.2024.10691696
  [3] Choudhary et al. (2025). Hybrid CNN-LSTM Model for EEG-Based Emotion
      Recognition: A Comparative Analysis Using DEAP and SEED Datasets.
      IEEE IC3IT 2025. DOI: 10.1109/IC3IT66137.2025.11341346
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 基础模块 ────────────────────────────────────────────────────────────────


class ChannelAttention(nn.Module):
    """通道注意力机制 — 为不同EEG通道分配重要性权重"""

    def __init__(self, n_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction),
            nn.ReLU(),
            nn.Linear(n_channels // reduction, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B, C, T] → [B, C, T]"""
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class SpatialAttention(nn.Module):
    """空间注意力机制 — 关注关键时间点"""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        """x: [B, C, T] → [B, C, T]"""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(attn))
        return x * attn


class AttentionLayer(nn.Module):
    """多头自注意力机制层"""

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return attended


class FreqBandFeatureExtractor(nn.Module):
    """多频段特征提取器 — 分别处理 Theta/Alpha/Beta/Gamma 频段"""

    def __init__(self, n_channels, band_width):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1, groups=n_channels),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
        )
        self.fc = nn.Linear(n_channels * band_width, 128)

    def forward(self, x):
        """x: [B, C, freq_bins] → [B, 128]"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ─── 电极‑矩形映射 ───────────────────────────────────────────────────────────


def build_electrode_map(n_channels=32, grid_h=9, grid_w=9):
    """将EEG电极位置映射到矩形网格（论文[2] Fig.3）

    对于 DEAP 32通道:  9×9 网格
    对于 SEED 62通道:  10×10 网格

    Returns:
        map_2d:   [grid_h, grid_w]  电极索引, 无效位置为 -1
    """
    if n_channels == 32:
        # DEAP 32通道 近似 9×9 布局
        mapping = torch.full((grid_h, grid_w), -1, dtype=torch.long)
        idx = 0
        for i in range(grid_h):
            for j in range(grid_w):
                if idx < n_channels:
                    mapping[i, j] = idx
                    idx += 1
        return mapping
    elif n_channels == 62:
        # SEED 62通道 近似 10×10 布局
        mapping = torch.full((10, 10), -1, dtype=torch.long)
        idx = 0
        for i in range(10):
            for j in range(10):
                if idx < n_channels:
                    mapping[i, j] = idx
                    idx += 1
        return mapping
    else:
        # 通用 fallback: 自动计算网格
        grid_size = math.ceil(math.sqrt(n_channels))
        mapping = torch.full((grid_size, grid_size), -1, dtype=torch.long)
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n_channels:
                    mapping[i, j] = idx
                    idx += 1
        return mapping


class ElectrodeMapping2D(nn.Module):
    """将 1D EEG 信号映射到 2D 矩形网格（论文[2]）"""

    def __init__(self, n_channels=32, grid_h=9, grid_w=9, n_timepoints=1280):
        super().__init__()
        self.n_channels = n_channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_timepoints = n_timepoints
        mapping = build_electrode_map(n_channels, grid_h, grid_w)
        self.register_buffer("mapping", mapping)

    def forward(self, x):
        """x: [B, C, T] → [B, 1, grid_h, grid_w, T] → [B, T, grid_h, grid_w]"""
        b, c, t = x.size()
        device = x.device
        grid = torch.zeros(b, self.grid_h, self.grid_w, t, device=device)
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                ch = self.mapping[i, j]
                if ch >= 0 and ch < c:
                    grid[:, i, j, :] = x[:, ch, :]
        return grid  # [B, H, W, T]


# ─── 主模型 ──────────────────────────────────────────────────────────────────


class EEGStressCNNLSTM(nn.Module):
    """混合 CNN-LSTM 模型 — EEG 压力/情绪识别

    架构（基于论文[1][2][3]）:
      CNN  →  空间/频率特征提取
      LSTM →  时序依赖建模
      Attn →  注意力加权
      Head →  回归(压力值) 或 分类(情绪标签)

    Args:
        n_channels:    EEG通道数 (DEAP:32, SEED:62)
        n_timepoints:  时间点数 (DEAP:1280, SEED:800)
        cnn_channels:  CNN层输出通道数列表
        lstm_hidden:   LSTM隐藏层大小
        lstm_layers:   LSTM层数
        dropout_rate:  Dropout比率
        num_classes:   输出维度 (回归=1, 分类=情绪类别数)
        task_type:     'regression' 或 'classification'
        use_2d_mapping: 是否使用电极‑矩形映射（论文[2]）
        n_freq_bands:   频带数量（差分熵特征使用）
    """

    def __init__(
        self,
        n_channels=32,
        n_timepoints=1280,
        cnn_channels=(64, 128, 256),
        cnn_kernel_sizes=(7, 5, 5),
        lstm_hidden=128,
        lstm_layers=2,
        lstm_bidirectional=True,
        lstm_dropout=0.3,
        dropout_rate=0.3,
        attention_heads=4,
        activation="relu",
        num_classes=1,
        task_type="regression",
        use_2d_mapping=False,
        n_freq_bands=4,
    ):
        super().__init__()

        # 激活函数选择
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(0.01),
        }
        act = act_map.get(activation, nn.ReLU)

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.task_type = task_type
        self.use_2d_mapping = use_2d_mapping
        self.n_freq_bands = n_freq_bands

        # ── 2D 电极映射（论文[2] 第三节B） ──
        if use_2d_mapping:
            grid_h = grid_w = math.ceil(math.sqrt(n_channels))
            self.electrode_map = ElectrodeMapping2D(n_channels, grid_h, grid_w, n_timepoints)
            cnn_in_channels = 1
            # 2D CNN 处理映射后的特征
            self.cnn_2d = nn.Sequential(
                nn.Conv2d(cnn_in_channels, cnn_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                nn.Conv2d(cnn_channels[0], cnn_channels[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_channels[0]),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            cnn_out = cnn_channels[0]
        else:
            self.electrode_map = None
            self.cnn_2d = None
            cnn_out = cnn_channels[-1]

        # ── 1D CNN 空间/频率特征提取（论文[2] 第三节C） ──
        self.cnn_layers = nn.ModuleList()
        for i in range(len(cnn_channels)):
            in_ch = n_channels if i == 0 else cnn_channels[i - 1]
            k = cnn_kernel_sizes[i] if i < len(cnn_kernel_sizes) else 3
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, cnn_channels[i], kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(cnn_channels[i]),
                    act(),
                    nn.Dropout(dropout_rate),
                )
            )

        # 时间池化降采样
        self.temporal_pool = nn.AdaptiveMaxPool1d(n_timepoints // 4)

        # 通道 + 空间注意力
        self.channel_attn = ChannelAttention(cnn_channels[-1])
        self.spatial_attn = SpatialAttention()

        # ── LSTM 时序建模（论文[2] 第三节D） ──
        lstm_input_size = cnn_out if use_2d_mapping else cnn_channels[-1]
        self.num_directions = 2 if lstm_bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # 多头注意力
        lstm_out_dim = lstm_hidden * self.num_directions
        self.attention = AttentionLayer(lstm_out_dim, num_heads=attention_heads)

        # ── 输出头 ──
        if task_type == "classification":
            # 分类头（论文[2] 第三节E）
            self.classifier = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(lstm_hidden, num_classes),
            )
            self.regressor = None
        else:
            # 回归头 — 输出连续压力值 [0,1]（论文[1]）
            self.regressor = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(lstm_hidden, num_classes),
                nn.Sigmoid(),
            )
            self.classifier = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """前向传播

        Args:
            x: [batch_size, n_channels, n_timepoints]  或  [batch_size, n_channels, n_timepoints, n_freq_bands]

        Returns:
            regression: [batch_size]  压力强度值 (0~1)
            classification: [batch_size, num_classes]  类别logits
        """
        # 多频段输入处理
        if x.dim() == 4:
            # [B, C, T, F] → 在频段维度上平均或拼接后处理
            b, c, t, f = x.size()
            x = x.view(b, c * f, t)

        # ── 2D 电极映射路径（论文[2]） ──
        if self.use_2d_mapping and self.electrode_map is not None:
            grid = self.electrode_map(x)  # [B, H, W, T]
            grid = grid.unsqueeze(1)      # [B, 1, H, W, T]
            # 分别处理每个时间片
            b, _, h, w, t = grid.size()
            cnn_out_list = []
            for ti in range(t):
                frame = grid[:, :, :, :, ti]  # [B, 1, H, W]
                feat = self.cnn_2d(frame)     # [B, C', 1, 1]
                cnn_out_list.append(feat.view(b, -1, 1))
            cnn_features = torch.cat(cnn_out_list, dim=2)  # [B, C', T]
        else:
            # ── 标准 1D CNN 路径 ──
            cnn_features = x
            for cnn_layer in self.cnn_layers:
                cnn_features = cnn_layer(cnn_features)

            # 注意力增强
            cnn_features = self.channel_attn(cnn_features)
            cnn_features = self.spatial_attn(cnn_features)

            # 时间池化
            cnn_features = self.temporal_pool(cnn_features)

        # ── LSTM 时序建模 ──
        lstm_input = cnn_features.transpose(1, 2)  # [B, T', C]
        lstm_output, _ = self.lstm(lstm_input)     # [B, T', H*2]

        # 注意力机制
        attended = self.attention(lstm_output)
        final_features = attended[:, -1, :]  # [B, H*2]

        # ── 输出 ──
        if self.task_type == "classification":
            return self.classifier(final_features)
        else:
            return self.regressor(final_features).squeeze(-1)


# ─── 频域模型 ────────────────────────────────────────────────────────────────


class FrequencyDomainEEGModel(nn.Module):
    """频域EEG压力/情绪识别模型

    基于论文[1][2]的频域特征分析:
      - 差分熵(DE)特征
      - 多频段能量特征 (Theta, Alpha, Beta, Gamma)
    """

    def __init__(
        self,
        n_channels=32,
        freq_bands=5,
        dropout_rate=0.3,
        task_type="regression",
        num_classes=1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.freq_bands = freq_bands

        out_dim = num_classes if task_type == "regression" else num_classes

        self.freq_extractor = nn.Sequential(
            nn.Linear(n_channels * freq_bands, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, out_dim),
            nn.Sigmoid() if task_type == "regression" else nn.Identity(),
        )

    def forward(self, freq_features):
        """freq_features: [batch_size, n_channels * freq_bands]"""
        out = self.freq_extractor(freq_features)
        return out.squeeze(-1)


# ─── 集成模型 ────────────────────────────────────────────────────────────────


class EnsembleEEGModel(nn.Module):
    """EEG模型集成 — 融合时域和频域特征（论文[1]）"""

    def __init__(self, n_channels=32, n_timepoints=1280, task_type="regression", num_classes=1):
        super().__init__()

        self.task_type = task_type
        self.num_classes = num_classes

        self.time_model = EEGStressCNNLSTM(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            task_type="regression" if task_type == "regression" else "classification",
            num_classes=num_classes,
        )
        self.freq_model = FrequencyDomainEEGModel(
            n_channels=n_channels,
            task_type="regression" if task_type == "regression" else "classification",
            num_classes=num_classes,
        )

        # 融合层
        fusion_in = 2 if task_type == "regression" else num_classes * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Sigmoid() if task_type == "regression" else nn.Identity(),
        )

    def forward(self, x_time, x_freq):
        time_pred = self.time_model(x_time)
        freq_pred = self.freq_model(x_freq)

        if time_pred.dim() == 0:
            time_pred = time_pred.unsqueeze(0)
        if freq_pred.dim() == 0:
            freq_pred = freq_pred.unsqueeze(0)

        time_pred = time_pred.unsqueeze(-1)
        freq_pred = freq_pred.unsqueeze(-1)
        fusion_input = torch.cat([time_pred, freq_pred], dim=-1)
        return self.fusion(fusion_input).squeeze(-1)


# ─── GAN数据增强模块（论文[3] 第三节3） ────────────────────────────────────────


class EEGGenerator(nn.Module):
    """生成器 — 生成合成EEG信号"""

    def __init__(self, latent_dim=100, n_channels=32, n_timepoints=1280):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_channels * n_timepoints),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.fc(z).view(-1, 32, 1280)


class EEGDiscriminator(nn.Module):
    """判别器 — 区分真实/合成EEG信号"""

    def __init__(self, n_channels=32, n_timepoints=1280):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
        )
        conv_out = 256 * (n_timepoints // 8)
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.conv(x)
        return self.fc(features.view(features.size(0), -1))


# ─── 差分熵(DE)特征提取（论文[2] 第三节B） ────────────────────────────────────


def compute_differential_entropy(eeg_signal, eps=1e-8):
    """计算差分熵(DE)特征

    DE(x) = 0.5 * log(2πeσ²)

    论文[2] 公式(2):
      h(X) = ½ log(2πeσ²)

    Args:
        eeg_signal: [n_channels, n_timepoints]  或  [batch, n_channels, n_timepoints]
        eps: 数值稳定性常数

    Returns:
        de_features: 差分熵值 [n_channels] 或 [batch, n_channels]
    """
    variance = eeg_signal.var(dim=-1, unbiased=False) + eps
    return 0.5 * torch.log(2 * math.pi * math.e * variance)


def extract_band_de_features(eeg_signal, fs=128):
    """提取多频段差分熵特征（论文[2] 第三节A‑B）

    频段划分:
      Theta:  4–8 Hz
      Alpha:  8–14 Hz
      Beta:  14–31 Hz
      Gamma: 31–45 Hz

    Args:
        eeg_signal: [batch, n_channels, n_timepoints]
        fs: 采样频率 (DEAP:128Hz, SEED:200Hz)

    Returns:
        band_de: [batch, n_channels, n_freq_bands]  各频段DE值
    """

    def _bandpass(signal, low, high, fs):
        """简易IIR带通滤波 (使用torch实现)"""
        nyquist = fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        # 使用有限差分近似带通滤波
        kernel_size = int(fs * 0.1) | 1  # 确保奇数
        kernel = torch.sin(torch.linspace(-math.pi, math.pi, kernel_size)) / (
            torch.linspace(-math.pi, math.pi, kernel_size) + 1e-8
        )
        kernel *= torch.hamming_window(kernel_size)
        # 频移实现带通
        center = (low + high) / 2
        t = torch.arange(kernel_size, device=signal.device)
        modulated = kernel * torch.cos(2 * math.pi * center / fs * t)
        modulated = modulated / modulated.sum()
        modulated = modulated.view(1, 1, -1).to(signal.device)
        return F.conv1d(signal, modulated.expand(signal.size(1), 1, -1), padding=kernel_size // 2, groups=signal.size(1))

    bands = [(4, 8), (8, 14), (14, 31), (31, 45)]
    band_features = []

    for low, high in bands:
        filtered = _bandpass(eeg_signal, low, high, fs)
        de = compute_differential_entropy(filtered)  # [B, C] or [C]
        if de.dim() == 1:
            de = de.unsqueeze(0)
        band_features.append(de.unsqueeze(-1))

    return torch.cat(band_features, dim=-1)  # [B, C, 4]


# ─── 工厂函数 ────────────────────────────────────────────────────────────────


def create_model(model_type="cnn_lstm", **kwargs):
    """创建EEG压力/情绪识别模型

    Args:
        model_type: 'cnn_lstm' | 'freq' | 'ensemble'
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    registry = {
        "cnn_lstm": EEGStressCNNLSTM,
        "freq": FrequencyDomainEEGModel,
        "ensemble": EnsembleEEGModel,
    }
    if model_type not in registry:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(registry.keys())}")
    return registry[model_type](**kwargs)


# ─── 模型测试 ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("EEG压力/情绪识别模型测试")
    print("=" * 60)

    # 测试1: 回归模型 (压力值)
    print("\n[测试1] 回归任务 — 连续压力值预测")
    model = EEGStressCNNLSTM(
        n_channels=32,
        n_timepoints=1280,
        task_type="regression",
        use_2d_mapping=False,
    ).to(device)
    x = torch.randn(16, 32, 1280).to(device)
    out = model(x)
    print(f"  输入: {x.shape} → 输出: {out.shape}, 范围: [{out.min():.3f}, {out.max():.3f}]")

    # 测试2: 分类模型 (情绪类别)
    print("\n[测试2] 分类任务 — 情绪类别识别")
    model_cls = EEGStressCNNLSTM(
        n_channels=32,
        n_timepoints=1280,
        task_type="classification",
        num_classes=3,
    ).to(device)
    out_cls = model_cls(x)
    print(f"  输入: {x.shape} → 输出: {out_cls.shape} (3类logits)")

    # 测试3: 2D电极映射 + 分类（论文[2]）
    print("\n[测试3] 2D电极映射 + 分类")
    model_2d = EEGStressCNNLSTM(
        n_channels=32,
        n_timepoints=1280,
        task_type="classification",
        num_classes=2,
        use_2d_mapping=True,
    ).to(device)
    out_2d = model_2d(x)
    print(f"  输入: {x.shape} → 输出: {out_2d.shape} (2类logits)")

    # 测试4: DE特征提取
    print("\n[测试4] 差分熵(DE)特征提取")
    de = compute_differential_entropy(x)
    print(f"  DE特征: {de.shape}")

    band_de = extract_band_de_features(x, fs=128)
    print(f"  多频段DE特征: {band_de.shape}")

    # 测试5: 频域模型
    print("\n[测试5] 频域模型")
    freq_model = FrequencyDomainEEGModel(n_channels=32, freq_bands=4).to(device)
    freq_in = torch.randn(16, 32 * 4).to(device)
    freq_out = freq_model(freq_in)
    print(f"  输入: {freq_in.shape} → 输出: {freq_out.shape}")

    # 测试6: 集成模型
    print("\n[测试6] 集成模型（时域+频域）")
    ensemble = EnsembleEEGModel(n_channels=32, n_timepoints=1280).to(device)
    ens_out = ensemble(x, freq_in)
    print(f"  输出: {ens_out.shape}")

    # 测试7: 2D映射构建
    print("\n[测试7] 电极‑矩形映射")
    map_32 = build_electrode_map(32)
    print(f"  DEAP 32通道映射: {map_32.shape}")
    print(f"  有效电极数: {(map_32 >= 0).sum().item()}")

    map_62 = build_electrode_map(62)
    print(f"  SEED 62通道映射: {map_62.shape}")
    print(f"  有效电极数: {(map_62 >= 0).sum().item()}")

    print("\n所有测试通过!")
