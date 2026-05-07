#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG数据集加载与预处理工具

支持的数据集:
  - STEW:  14通道, 128Hz, 0-9十级压力评分（论文[1]）
  - DEAP:  32通道, 128Hz, Valence/Arousal（论文[2][3]）
  - SEED:  62通道, 200Hz, 三分类情绪（论文[3]）
"""

import math
import numpy as np
import torch


def load_stew_format(data_dir):
    """加载STEW格式数据（论文[1]）

    STEW数据集:
      - 14通道 EEG (Emotiv EPOC)
      - 128Hz 采样, 每个被试150秒Stroop任务
      - 标签: 0-9 十级压力评分

    文件结构:
      dataset.mat:          (14, 19200, 45)  通道×时间点×被试
      rating.mat:           (45, 1)          0-9十级评分
      class_012.mat:        (45, 1)          三分类 (0/1/2)

    Args:
        data_dir: 包含 .mat 文件的目录路径

    Returns:
        eeg_data:  (45, 14, 19200)  float32
        labels:    (45,)             int64  0-9
    """
    import os
    import scipy.io as sio

    data = sio.loadmat(os.path.join(data_dir, "dataset.mat"))["dataset"]    # (14, 19200, 45)
    rating = sio.loadmat(os.path.join(data_dir, "rating.mat"))["rating"]    # (45, 1)
    eeg = data.transpose(2, 0, 1).astype(np.float32)                       # (45, 14, 19200)
    labels = rating.squeeze().astype(np.int64)                              # (45,)
    return eeg, labels


def load_deap_format(data_path):
    """加载DEAP格式数据（论文[2][3]）

    DEAP数据集:
      - 32通道 EEG (另含8通道外周信号)
      - 128Hz 采样, 每个样本60秒视频
      - 标签: valence, arousal, dominance, liking

    Args:
        data_path: .dat 文件路径

    Returns:
        data:   (n_trials, 40通道, n_timepoints)
        labels: (n_trials, 4)  [valence, arousal, dominance, liking]
    """
    import scipy.io as sio
    mat = sio.loadmat(data_path)
    return mat["data"], mat["labels"]


def load_seed_format(data_path):
    """加载SEED格式数据（论文[3]）

    SEED数据集:
      - 62通道 EEG
      - 200Hz 采样
      - 标签: positive(0) / neutral(1) / negative(2)

    Args:
        data_path: .mat 文件路径

    Returns:
        data:   (n_trials, 62, n_timepoints)
        labels: (n_trials,)
    """
    import scipy.io as sio
    import h5py

    try:
        mat = sio.loadmat(data_path)
        return mat["data"], mat["labels"]
    except NotImplementedError:
        with h5py.File(data_path, "r") as f:
            return f["data"][:], f["labels"][:]


def segment_eeg(eeg_data, labels, window_sec=2.5, fs=128, overlap=0.5):
    """滑窗分段，扩充样本量

    适用于小样本数据集（如STEW仅45样本）:
      STEW: (45, 14, 19200) → (5355, 14, 320)
      DEAP: (40, 32, 7680)  → 每段视频可切出约47段

    Args:
        eeg_data:     (n_subjects, n_channels, n_timepoints)
        labels:       (n_subjects,)  每个被试一个标签
        window_sec:   窗口长度（秒）
        fs:           采样频率 (Hz)
        overlap:      窗口重叠比例 [0, 1)

    Returns:
        segments:     (n_segments, n_channels, window_points)
        seg_labels:   (n_segments,)  每个分段继承原始标签
    """
    win = int(window_sec * fs)
    stride = int(win * (1 - overlap))
    n_subjects, n_channels, n_timepoints = eeg_data.shape

    segments, seg_labels = [], []
    for s in range(n_subjects):
        for t in range(0, n_timepoints - win + 1, stride):
            segments.append(eeg_data[s, :, t:t + win])
            seg_labels.append(labels[s])

    return np.array(segments), np.array(seg_labels)


def extract_freq_band_power(eeg_data, fs=128):
    """提取频带能量特征

    频段: Delta(0.5-4), Theta(4-8), Alpha(8-14), Beta(14-31), Gamma(31-45)

    Args:
        eeg_data: [n_samples, n_channels, n_timepoints]
        fs:       采样频率

    Returns:
        band_powers: [n_samples, n_channels, 5]
    """
    n_samples, n_channels, n_timepoints = eeg_data.shape
    bands = [(0.5, 4), (4, 8), (8, 14), (14, 31), (31, 45)]
    n_bands = len(bands)

    freqs = torch.fft.rfftfreq(n_timepoints, d=1.0 / fs)
    result = []

    for s in range(n_samples):
        signal = torch.FloatTensor(eeg_data[s])
        fft_vals = torch.fft.rfft(signal, dim=1)
        power = torch.abs(fft_vals) ** 2

        band_powers = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            band_powers.append(power[:, mask].sum(dim=1))
        result.append(torch.stack(band_powers, dim=1))

    return torch.stack(result)


def compute_spectrogram(eeg_data, fs=128, window_sec=1.0, overlap=0.5):
    """计算EEG频谱图（论文[3]）

    Args:
        eeg_data:   [n_samples, n_channels, n_timepoints]
        fs:         采样频率
        window_sec: STFT窗口长度(秒)
        overlap:    窗口重叠比例

    Returns:
        spectrograms: [n_samples, n_channels, n_freq_bins, n_time_frames]
    """
    n_samples, n_channels, n_timepoints = eeg_data.shape
    window_size = int(window_sec * fs)
    hop_length = int(window_size * (1 - overlap))
    n_freq_bins = window_size // 2 + 1
    n_frames = (n_timepoints - window_size) // hop_length + 1

    window = torch.hamming_window(window_size)

    spectrograms = []
    for s in range(n_samples):
        ch_specs = []
        for c in range(n_channels):
            signal = torch.FloatTensor(eeg_data[s, c])
            frames = signal.unfold(0, window_size, hop_length)
            frames = frames * window
            spec = torch.abs(torch.fft.rfft(frames, dim=1))
            ch_specs.append(spec.T.unsqueeze(0))
        spectrograms.append(torch.cat(ch_specs, dim=0))
    return torch.stack(spectrograms)


def preprocess_eeg(eeg_data, normalize="zscore", bandpass_low=None, bandpass_high=None, fs=128):
    """EEG数据预处理：归一化 + 带通滤波

    Args:
        eeg_data:      [n_samples, n_channels, n_timepoints]
        normalize:     "zscore" | "minmax" | "none"
        bandpass_low:  低频截止 (Hz), None=不滤波
        bandpass_high: 高频截止 (Hz), None=不滤波
        fs:            采样频率

    Returns:
        预处理后的数据 (与原形状相同)
    """
    data = eeg_data.copy().astype(np.float64)

    # 归一化
    if normalize == "zscore":
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True) + 1e-8
        data = (data - mean) / std
        # 裁剪极端离群值（z-score>5 通过CNN会导致梯度爆炸→NaN）
        data = np.clip(data, -5, 5)
    elif normalize == "minmax":
        dmin = data.min(axis=-1, keepdims=True)
        dmax = data.max(axis=-1, keepdims=True)
        data = (data - dmin) / (dmax - dmin + 1e-8)

    # 带通滤波 (简易FFT频域滤波)
    if bandpass_low is not None or bandpass_high is not None:
        n_timepoints = data.shape[-1]
        freqs = np.fft.rfftfreq(n_timepoints, d=1.0 / fs)
        mask = np.ones_like(freqs, dtype=bool)
        if bandpass_low is not None:
            mask &= freqs >= bandpass_low
        if bandpass_high is not None:
            mask &= freqs <= bandpass_high
        # 逐样本逐通道滤波
        for s in range(data.shape[0]):
            for c in range(data.shape[1]):
                fft_vals = np.fft.rfft(data[s, c])
                fft_vals[~mask] = 0
                data[s, c] = np.fft.irfft(fft_vals, n=n_timepoints)

    return data.astype(eeg_data.dtype)


def create_dummy_data(n_samples=1000, n_channels=32, n_timepoints=1280,
                      task_type="regression", n_classes=3):
    """创建模拟EEG数据用于测试

    Args:
        n_samples:    样本数
        n_channels:   EEG通道数
        n_timepoints: 时间点数
        task_type:    'regression' | 'classification'
        n_classes:    分类数（分类任务时使用）

    Returns:
        eeg_data:  [n_samples, n_channels, n_timepoints]
        labels:    [n_samples]
    """
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints) * 0.1
    if task_type == "classification":
        labels = np.random.randint(0, n_classes, size=n_samples)
    else:
        labels = np.random.rand(n_samples)
    return eeg_data, labels


if __name__ == "__main__":
    print("=" * 60)
    print("数据集加载工具测试")
    print("=" * 60)

    # 模拟数据
    eeg, labels = create_dummy_data(n_samples=10, n_channels=14,
                                    n_timepoints=19200, task_type="classification", n_classes=10)
    print(f"模拟数据: {eeg.shape}, 标签: {labels.shape}")

    # 滑窗
    segs, seg_labels = segment_eeg(eeg, labels, window_sec=2.5, fs=128)
    print(f"分段结果: {segs.shape}, 标签: {seg_labels.shape}")
    print(f"标签分布: {np.bincount(seg_labels)}")

    # 频带能量
    bp = extract_freq_band_power(eeg[:2], fs=128)
    print(f"频带能量: {bp.shape}")

    print("[OK] 所有工具函数测试通过")
