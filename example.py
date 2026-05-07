#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG压力/情绪识别 — 使用示例

基于以下论文:
  [1] Hybrid CNN–LSTM Model For Continuous Stress Quantification Via
      Emotion-Valence Mapping On EEG Signals (Chaudhari & Shrivastava, 2026)
  [2] EEG Emotion Recognition Based on CNN+LSTM (Zhu, Song & Li, 2024)
  [3] Hybrid CNN-LSTM Model for EEG-Based Emotion Recognition:
      A Comparative Analysis Using DEAP and SEED Datasets (Choudhary et al., 2025)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import (
    EEGStressCNNLSTM,
    FrequencyDomainEEGModel,
    EnsembleEEGModel,
    compute_differential_entropy,
    extract_band_de_features,
    build_electrode_map,
)
from trainer import (
    EEGStressTrainer,
    EEGStressDataset,
    EEGDataAugmentation,
    cross_validate,
)
from dataset_transform import (
    create_dummy_data,
    extract_freq_band_power,
    compute_spectrogram,
)
from torch.utils.data import DataLoader


def demo_basic():
    """基础演示 — 创建模型、模拟数据、前向传播"""
    print("=" * 60)
    print("EEG压力/情绪识别模型 — 基础演示")
    print("=" * 60)

    # 1. 创建模型
    print("\n[1] 创建回归模型 (连续压力值输出)")
    model = EEGStressCNNLSTM(
        n_channels=32,
        n_timepoints=1280,
        cnn_channels=[64, 128, 256],
        lstm_hidden=128,
        lstm_layers=2,
        dropout_rate=0.3,
        task_type="regression",
    )
    print(f"   参数总量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 模拟数据
    print("\n[2] 生成模拟EEG数据")
    eeg_data, stress_labels = create_dummy_data(
        n_samples=1000, n_channels=32, n_timepoints=1280, task_type="regression"
    )
    print(f"   EEG数据: {eeg_data.shape}")
    print(f"   标签: [{stress_labels.min():.3f}, {stress_labels.max():.3f}]")

    # 3. 数据分割
    print("\n[3] 数据分割 (train/val/test = 60/20/20)")
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        eeg_data, stress_labels, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"   训练: {len(X_train)} | 验证: {len(X_val)} | 测试: {len(X_test)}")

    # 4. 数据加载器
    train_dataset = EEGStressDataset(
        X_train, y_train, task_type="regression",
        transform=EEGDataAugmentation(noise_std=0.01, channel_dropout_prob=0.1)
    )
    val_dataset = EEGStressDataset(X_val, y_val, task_type="regression")
    test_dataset = EEGStressDataset(X_test, y_test, task_type="regression")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 5. 单样本测试
    print("\n[4] 单样本前向传播")
    model.eval()
    sample_data, sample_label = train_dataset[0]
    with torch.no_grad():
        prediction = model(sample_data.unsqueeze(0))
    print(f"   预测: {prediction.item():.4f} | 真实: {sample_label:.4f}")

    # 6. 可视化
    print("\n[5] 可视化EEG信号和预测结果")
    visualize_eeg(sample_data, sample_label, prediction.item())

    return model, (train_loader, val_loader, test_loader)


def demo_classification():
    """分类演示 — 情绪类别识别（论文[2]）"""
    print("\n" + "=" * 60)
    print("情绪分类演示 (Valence/Arousal二分类)")
    print("=" * 60)

    # 创建分类模型
    model = EEGStressCNNLSTM(
        n_channels=32,
        n_timepoints=1280,
        task_type="classification",
        num_classes=2,
        use_2d_mapping=True,  # 使用2D电极映射（论文[2]）
    )
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟数据
    eeg_data, labels = create_dummy_data(
        n_samples=500, task_type="classification", n_classes=2
    )
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        eeg_data, labels, test_size=0.2, random_state=42
    )

    train_dataset = EEGStressDataset(X_train, y_train, task_type="classification")
    val_dataset = EEGStressDataset(X_val, y_val, task_type="classification")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    trainer = EEGStressTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={"epochs": 5, "log_dir": "./logs_classification", "model_save_dir": "./models_classification"},
        task_type="classification",
    )
    trainer.train()

    print("分类演示完成!")
    return model


def demo_de_features():
    """差分熵(DE)特征演示（论文[2] 第三节B）"""
    print("\n" + "=" * 60)
    print("差分熵(DE)特征提取演示")
    print("=" * 60)

    eeg_data, _ = create_dummy_data(n_samples=4, n_channels=32, n_timepoints=1280)
    x = torch.FloatTensor(eeg_data)

    # 计算DE特征
    de = compute_differential_entropy(x)
    print(f"DE特征形状: {de.shape}  (batch, channels)")

    # 多频段DE特征
    band_de = extract_band_de_features(x, fs=128)
    print(f"多频段DE特征: {band_de.shape}  (batch, channels, bands)")
    print(f"频段: Theta(4-8), Alpha(8-14), Beta(14-31), Gamma(31-45)")

    # 频带能量特征
    band_power = extract_freq_band_power(x.numpy(), fs=128)
    print(f"频带能量: {band_power.shape} (batch, channels, 5 bands)")
    print(f"频段: Delta(0.5-4), Theta(4-8), Alpha(8-14), Beta(14-31), Gamma(31-45)")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # DE热图
    im1 = axes[0, 0].imshow(de.numpy(), aspect="auto", cmap="viridis")
    axes[0, 0].set_title("差分熵(DE) — 各通道")
    axes[0, 0].set_xlabel("通道")
    axes[0, 0].set_ylabel("样本")
    plt.colorbar(im1, ax=axes[0, 0])

    # 频段DE
    for i in range(4):
        axes[0, 1].plot(band_de[0, :, i].numpy(), label=["Theta", "Alpha", "Beta", "Gamma"][i])
    axes[0, 1].set_title("多频段DE (样本0)")
    axes[0, 1].set_xlabel("通道")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 频谱图
    spec = compute_spectrogram(eeg_data[:1], fs=128)
    spec_db = 10 * np.log10(spec.numpy().squeeze(0).mean(axis=-1) + 1e-10)
    im2 = axes[1, 0].imshow(spec_db, aspect="auto", cmap="magma", origin="lower")
    axes[1, 0].set_title("频谱图 (平均)")
    axes[1, 0].set_xlabel("频率 (Hz)")
    axes[1, 0].set_ylabel("通道")
    plt.colorbar(im2, ax=axes[1, 0])

    # 电极映射
    mapping = build_electrode_map(32)
    map_display = mapping.numpy()
    axes[1, 1].imshow(map_display, cmap="tab20", interpolation="nearest")
    for i in range(map_display.shape[0]):
        for j in range(map_display.shape[1]):
            val = map_display[i, j]
            axes[1, 1].text(j, i, f"{int(val)}" if val >= 0 else "",
                            ha="center", va="center", fontsize=8)
    axes[1, 1].set_title("电极→矩形映射 (DEAP 32通道)")

    plt.tight_layout()
    plt.savefig("./de_features_demo.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("DE特征可视化已保存到 ./de_features_demo.png")


def demo_ensemble():
    """集成模型演示（论文[1] — 时域+频域融合）"""
    print("\n" + "=" * 60)
    print("集成模型演示 (时域+频域融合)")
    print("=" * 60)

    model = EnsembleEEGModel(
        n_channels=32,
        n_timepoints=1280,
        task_type="regression",
        num_classes=1,
    )
    print(f"集成模型参数: {sum(p.numel() for p in model.parameters()):,}")

    eeg_data, labels = create_dummy_data(n_samples=10, task_type="regression")
    freq_data = extract_freq_band_power(eeg_data, fs=128)
    freq_features = freq_data.view(10, -1)

    x_time = torch.FloatTensor(eeg_data)
    x_freq = torch.FloatTensor(freq_features)

    model.eval()
    with torch.no_grad():
        output = model(x_time, x_freq)
    print(f"集成输出: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")


def demo_cross_validation():
    """K折交叉验证演示（论文[3]）"""
    print("\n" + "=" * 60)
    print("交叉验证演示 (3折)")
    print("=" * 60)

    eeg_data, labels = create_dummy_data(n_samples=100, task_type="regression")

    def model_factory():
        return EEGStressCNNLSTM(task_type="regression")

    fold_results, avg_metrics = cross_validate(
        model_factory=model_factory,
        eeg_data=eeg_data,
        labels=labels,
        n_folds=3,
        batch_size=16,
        task_type="regression",
        config={"epochs": 3, "patience": 5},
        seed=42,
    )
    print("交叉验证完成!")


def visualize_eeg(eeg_sample, true_label, predicted_label):
    """可视化EEG样本和预测结果"""
    n_channels_to_show = min(8, eeg_sample.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # EEG信号
    time_axis = np.arange(eeg_sample.shape[1])
    for i in range(n_channels_to_show):
        axes[0].plot(time_axis, eeg_sample[i].numpy() + i * 2, label=f"通道 {i+1}")
    axes[0].set_xlabel("时间点")
    axes[0].set_ylabel("振幅 (偏移)")
    axes[0].set_title("EEG信号 (前8个通道)")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 预测结果对比
    labels = ["真实值", "预测值"]
    values = [true_label, predicted_label]
    colors = ["skyblue", "lightcoral"]
    bars = axes[1].bar(labels, values, color=colors, alpha=0.7)
    axes[1].set_ylabel("压力强度")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("压力识别结果")
    for bar, v in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom")
    error = abs(true_label - predicted_label)
    axes[1].text(0.5, 0.95, f"绝对误差: {error:.3f}",
                 transform=axes[1].transAxes, ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    plt.tight_layout()
    plt.savefig("./sample_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("可视化已保存: ./sample_visualization.png")


def demo_training_pipeline():
    """快速训练Pipeline演示（简化版）"""
    print("\n" + "=" * 60)
    print("训练Pipeline演示 (简化版)")
    print("=" * 60)

    model = EEGStressCNNLSTM()
    eeg_data, stress_labels = create_dummy_data(n_samples=200, task_type="regression")

    split_idx = int(0.8 * len(eeg_data))
    X_train, X_val = eeg_data[:split_idx], eeg_data[split_idx:]
    y_train, y_val = stress_labels[:split_idx], stress_labels[split_idx:]

    train_dataset = EEGStressDataset(
        X_train, y_train, task_type="regression",
        transform=EEGDataAugmentation(noise_std=0.01)
    )
    val_dataset = EEGStressDataset(X_val, y_val, task_type="regression")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    trainer = EEGStressTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            "epochs": 3,
            "learning_rate": 1e-3,
            "log_dir": "./demo_logs",
            "model_save_dir": "./demo_models",
        },
        task_type="regression",
    )
    print("开始训练演示 (3 epochs)...")
    trainer.train()
    print("训练演示完成!")


if __name__ == "__main__":
    # 基础演示
    demo_basic()

    # 分类演示
    demo_classification()

    # DE特征演示
    demo_de_features()

    # 集成模型演示
    demo_ensemble()

    # 交叉验证演示
    demo_cross_validation()

    # 询问是否运行完整训练
    response = input("\n是否运行完整的训练Pipeline演示? (y/N): ")
    if response.lower() == "y":
        demo_training_pipeline()

    print("\n所有演示完成!")
