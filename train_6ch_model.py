#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6通道EEG压力分类模型训练脚本
2
基于 STEW 数据集，选取6个与W8放大器最匹配的通道：
  FC5, P7, O1, O2, P8, FC6

用法:
  python train_6ch_model.py
  python train_6ch_model.py --config my_config.yaml
"""

import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from model import EEGStressCNNLSTM
from trainer import EEGStressDataset, EEGStressTrainer, EEGDataAugmentation
from dataset_transform import load_stew_format, segment_eeg, preprocess_eeg


# STEW 14通道 → 选中的6通道索引
# Emotiv EPOC 14通道顺序: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4
SELECTED_CHANNEL_INDICES = [3, 5, 6, 7, 8, 10]  # FC5, P7, O1, O2, P8, FC6
CHANNEL_NAMES = ["FC5", "P7", "O1", "O2", "P8", "FC6"]

# STEW 0-9 十级评分 → 3分类映射
#   低压力: 0-3 → 0
#   中压力: 4-6 → 1
#   高压力: 7-9 → 2
STEW_TO_3CLASS = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1, 5: 1, 6: 1,
    7: 2, 8: 2, 9: 2,
}


def map_labels_to_3class(labels):
    """将STEW 0-9标签映射为3分类标签"""
    return np.array([STEW_TO_3CLASS[l] for l in labels], dtype=np.int64)


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="6通道EEG压力分类训练")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    split_cfg = cfg["split"]
    dataset_cfg = cfg["dataset"]
    paths_cfg = cfg["paths"]

    print("=" * 60)
    print(f"{cfg['dataset']['name']} 数据集 6通道训练")
    print(f"选中的通道: {CHANNEL_NAMES}")
    print(f"分类数: {cfg['num_classes']}")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device_name}")
    print("=" * 60)

    # ─── 1. 加载数据 ──────────────────────────────────────────
    eeg, labels = load_stew_format(dataset_cfg["data_dir"])
    print(f"\n原始数据: {eeg.shape}")

    # ─── 2. 选择6通道 ──────────────────────────────────────────
    eeg = eeg[:, SELECTED_CHANNEL_INDICES, :]
    print(f"6通道选择后: {eeg.shape}  {CHANNEL_NAMES}")

    # ─── 3. 预处理 ─────────────────────────────────────────────
    preproc = cfg.get("preprocessing", {})
    eeg = preprocess_eeg(
        eeg,
        normalize=preproc.get("normalize", "zscore"),
        bandpass_low=preproc.get("bandpass_low"),
        bandpass_high=preproc.get("bandpass_high"),
        fs=preproc.get("fs", dataset_cfg.get("fs", 128)),
    )

    # ─── 4. 滑窗分段 ───────────────────────────────────────────
    segs, seg_labels = segment_eeg(
        eeg, labels,
        window_sec=preproc.get("window_sec", 2.5),
        fs=preproc.get("fs", dataset_cfg.get("fs", 128)),
        overlap=preproc.get("overlap", 0.0),
    )
    print(f"\n分段后: {segs.shape}")
    print(f"标签分布: {np.bincount(seg_labels)}")

    # ─── 5. 划分 train/val/test ────────────────────────────────
    val_test_ratio = 1 - split_cfg["train_ratio"]
    val_ratio_in_val_test = split_cfg["val_ratio"] / (split_cfg["val_ratio"] + split_cfg["test_ratio"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        segs, seg_labels, test_size=val_test_ratio,
        random_state=split_cfg["random_seed"], stratify=seg_labels,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_ratio_in_val_test,
        random_state=split_cfg["random_seed"], stratify=y_tr,
    )
    print(f"\n划分: 训练 {X_tr.shape[0]} | 验证 {X_val.shape[0]} | 测试 {X_te.shape[0]}")

    # ─── 6. WeightedRandomSampler ───────────────────────────────
    WEIGHT_CAP_MULTIPLIER = 5
    class_counts = np.bincount(y_tr, minlength=cfg["num_classes"])
    raw_weights = 1.0 / (class_counts + 1e-8)
    median_weight = np.median(raw_weights[class_counts > 0])
    capped_weights = np.minimum(raw_weights, median_weight * WEIGHT_CAP_MULTIPLIER)
    sample_weights = torch.DoubleTensor(capped_weights[y_tr])
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # ─── 7. DataLoader ──────────────────────────────────────────
    tr_ds = EEGStressDataset(
        X_tr, y_tr, task_type=cfg["task_type"],
        transform=EEGDataAugmentation(
            noise_std=aug_cfg["noise_std"],
            channel_dropout_prob=aug_cfg["channel_dropout_prob"],
            time_mask_prob=aug_cfg["time_mask_prob"],
            time_mask_size=aug_cfg["time_mask_size"],
        ),
    )
    val_ds = EEGStressDataset(X_val, y_val, task_type=cfg["task_type"])
    te_ds = EEGStressDataset(X_te, y_te, task_type=cfg["task_type"])

    tr_loader = DataLoader(tr_ds, batch_size=train_cfg["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    # ─── 8. 创建6通道模型 ──────────────────────────────────────
    model = EEGStressCNNLSTM(
        n_channels=6,  # 6 通道！
        n_timepoints=model_cfg["n_timepoints"],
        cnn_channels=model_cfg["cnn_channels"],
        cnn_kernel_sizes=model_cfg.get("cnn_kernel_sizes", (7, 5, 5)),
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        lstm_bidirectional=model_cfg.get("lstm_bidirectional", True),
        lstm_dropout=model_cfg.get("lstm_dropout", model_cfg["dropout_rate"]),
        dropout_rate=model_cfg["dropout_rate"],
        attention_heads=model_cfg.get("attention_heads", 4),
        activation=model_cfg.get("activation", "relu"),
        task_type=cfg["task_type"],
        num_classes=cfg["num_classes"],
        use_2d_mapping=False,
    )
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ─── 9. 训练（保存目录改为 6ch） ───────────────────────────
    log_dir = paths_cfg["log_dir"].rstrip("/\\") + "_6ch"
    model_save_dir = paths_cfg["model_save_dir"].rstrip("/\\") + "_6ch"

    vis_cfg = cfg.get("visdom", {})
    trainer = EEGStressTrainer(
        model=model,
        train_loader=tr_loader,
        val_loader=val_loader,
        test_loader=te_loader,
        config={
            "epochs": train_cfg["epochs"],
            "optimizer": train_cfg.get("optimizer", "adamw"),
            "learning_rate": train_cfg["learning_rate"],
            "weight_decay": train_cfg["weight_decay"],
            "gradient_clip": train_cfg["gradient_clip"],
            "lr_scheduler": train_cfg["lr_scheduler"],
            "warmup_epochs": train_cfg.get("warmup_epochs", 5),
            "label_smoothing": train_cfg.get("label_smoothing", 0.0),
            "class_weight": train_cfg.get("class_weight", "none"),
            "mixed_precision": train_cfg.get("mixed_precision", False),
            "seed": train_cfg.get("seed", 42),
            "patience": train_cfg["patience"],
            "min_delta": train_cfg["min_delta"],
            "log_dir": log_dir,
            "model_save_dir": model_save_dir,
            "visdom_enabled": vis_cfg.get("enabled", False),
            "visdom_server": vis_cfg.get("server", "http://localhost"),
            "visdom_port": vis_cfg.get("port", 707),
            "visdom_env": vis_cfg.get("env", "eeg_stress"),
        },
        task_type=cfg["task_type"],
    )
    trainer.train()
    print("\n6通道模型训练完成!")
    print(f"模型保存路径: {model_save_dir}/best_model.pth")


if __name__ == "__main__":
    main()
