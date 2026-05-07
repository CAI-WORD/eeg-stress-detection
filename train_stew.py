#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEW数据集 — 10分类训练脚本

用法:
  python train_stew.py                         # 使用默认 config.yaml
  python train_stew.py --config my_config.yaml  # 使用自定义配置
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import EEGStressCNNLSTM
from trainer import EEGStressDataset, EEGStressTrainer, EEGDataAugmentation
from dataset_transform import load_stew_format, segment_eeg, preprocess_eeg


def load_config(config_path="config.yaml"):
    """加载 YAML 配置文件"""
    # 如果当前目录下找不到配置文件，尝试在脚本所在目录下查找
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"找不到配置文件: {config_path} (也在 {script_dir} 中尝试过)")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="STEW 10分类训练")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # ─── 加载配置 ────────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]
    split_cfg = cfg["split"]
    dataset_cfg = cfg["dataset"]
    paths_cfg = cfg["paths"]

    print("=" * 60)
    print(f"{cfg['dataset']['name']} 数据集 {cfg['task_type']} 训练")
    print(f"分类数: {cfg['num_classes']}")
    print(f"模型: CNN-LSTM, {model_cfg['cnn_channels']} → LSTM({model_cfg['lstm_hidden']})")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device_name} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU'})")
    print(f"Epochs: {train_cfg['epochs']}, Batch: {train_cfg['batch_size']}, LR: {train_cfg['learning_rate']}")
    print("=" * 60)

    # ─── 1. 加载数据 ──────────────────────────────────────────────────────────────
    eeg, labels = load_stew_format(dataset_cfg["data_dir"])
    print(f"\n原始数据: {eeg.shape}")
    print(f"标签分布(0-9): {np.bincount(labels)}")

    # ─── 2. 预处理 ────────────────────────────────────────────────────────────────
    preproc = cfg.get("preprocessing", {})
    eeg = preprocess_eeg(
        eeg,
        normalize=preproc.get("normalize", "zscore"),
        bandpass_low=preproc.get("bandpass_low"),
        bandpass_high=preproc.get("bandpass_high"),
        fs=preproc.get("fs", dataset_cfg.get("fs", 128)),
    )
    print(f"预处理后: {eeg.shape}")

    # ─── 3. 滑窗分段 ──────────────────────────────────────────────────────────────
    segs, seg_labels = segment_eeg(
        eeg, labels,
        window_sec=preproc.get("window_sec", dataset_cfg.get("window_sec", 2.5)),
        fs=preproc.get("fs", dataset_cfg.get("fs", 128)),
        overlap=preproc.get("overlap", dataset_cfg.get("overlap", 0.5)),
    )
    print(f"\n分段后: {segs.shape}")
    print(f"标签分布: {np.bincount(seg_labels)}")

    # ─── 4. 划分 ──────────────────────────────────────────────────────────────────
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

    # ─── 5. DataLoader ────────────────────────────────────────────────────────────
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

    tr_loader = DataLoader(tr_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    # ─── 6. 模型 ──────────────────────────────────────────────────────────────────
    model = EEGStressCNNLSTM(
        n_channels=model_cfg["n_channels"],
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
        use_2d_mapping=model_cfg["use_2d_mapping"],
    )
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ─── 7. 训练 ──────────────────────────────────────────────────────────────────
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
            "log_dir": paths_cfg["log_dir"],
            "model_save_dir": paths_cfg["model_save_dir"],
            # Visdom
            "visdom_enabled": vis_cfg.get("enabled", False),
            "visdom_server": vis_cfg.get("server", "http://localhost"),
            "visdom_port": vis_cfg.get("port", 707),
            "visdom_env": vis_cfg.get("env", "eeg_stress"),
        },
        task_type=cfg["task_type"],
    )
    trainer.train()
    print("\n训练完成!")


if __name__ == "__main__":
    main()
