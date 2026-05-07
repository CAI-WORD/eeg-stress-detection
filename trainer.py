#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG压力/情绪识别 — 训练器与数据处理

基于以下论文实现:
  [1] Chaudhari & Shrivastava (2026). Hybrid CNN–LSTM Model For Continuous
      Stress Quantification Via Emotion-Valence Mapping On EEG Signals.
  [2] Zhu, Song & Li (2024). EEG Emotion Recognition Based on CNN+LSTM.
      IEEE CCSSTA 2024.
  [3] Choudhary et al. (2025). Hybrid CNN-LSTM Model for EEG-Based Emotion
      Recognition: A Comparative Analysis Using DEAP and SEED Datasets.
      IEEE IC3IT 2025.
"""

import os
import time
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    from visdom import Visdom
    _VISDOM_AVAILABLE = True
except ImportError:
    _VISDOM_AVAILABLE = False

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import KFold

from model import EEGStressCNNLSTM, compute_differential_entropy, extract_band_de_features
from dataset_transform import (
    create_dummy_data,
    extract_freq_band_power,
    compute_spectrogram,
)


# ─── 数据集 ──────────────────────────────────────────────────────────────────


class EEGStressDataset(Dataset):
    """EEG压力/情绪识别数据集

    支持:
      - 回归任务 (压力值)
      - 分类任务 (情绪标签)
      - 多频段DE特征（论文[2]）
      - 数据增强
    """

    def __init__(
        self,
        eeg_data,
        labels,
        task_type="regression",
        transform=None,
        extract_de=False,
        fs=128,
    ):
        """
        Args:
            eeg_data:     EEG数据 [n_samples, n_channels, n_timepoints]
            labels:       标签 — 回归[0~1] / 分类[整数]
            task_type:    'regression' | 'classification'
            transform:    数据增强对象
            extract_de:   是否提取差分熵特征（论文[2]）
            fs:           采样频率
        """
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.FloatTensor(labels) if task_type == "regression" else torch.LongTensor(labels)
        self.transform = transform
        self.task_type = task_type
        self.extract_de = extract_de
        self.fs = fs

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]
        label = self.labels[idx]

        # 差分熵特征（论文[2]）
        if self.extract_de:
            de_feat = compute_differential_entropy(eeg_sample.unsqueeze(0), eps=1e-8)
            de_feat = de_feat.squeeze(0)  # [n_channels]
            band_de = extract_band_de_features(eeg_sample.unsqueeze(0), fs=self.fs)
            band_de = band_de.squeeze(0)  # [n_channels, n_bands]
            return eeg_sample, (label, de_feat, band_de)

        # 数据增强
        if self.transform:
            eeg_sample = self.transform(eeg_sample)

        return eeg_sample, label


class MultiModalEEGDataset(Dataset):
    """多模态EEG数据集 — 同时返回时域和频域特征（论文[1]集成模型）"""

    def __init__(self, eeg_data, labels, freq_features=None, task_type="regression"):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.FloatTensor(labels) if task_type == "regression" else torch.LongTensor(labels)
        self.freq_features = torch.FloatTensor(freq_features) if freq_features is not None else None
        self.task_type = task_type

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx], self.freq_features[idx] if self.freq_features is not None else torch.zeros(1)


# ─── 数据增强 ────────────────────────────────────────────────────────────────


class EEGDataAugmentation:
    """EEG数据增强（论文[3]）"""

    def __init__(
        self,
        noise_std=0.01,
        channel_dropout_prob=0.1,
        time_mask_prob=0.05,
        time_mask_size=20,
    ):
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob
        self.time_mask_prob = time_mask_prob
        self.time_mask_size = time_mask_size

    def __call__(self, eeg_sample):
        """应用数据增强

        Args:
            eeg_sample: [n_channels, n_timepoints]
        """
        # 高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(eeg_sample) * self.noise_std
            eeg_sample = eeg_sample + noise

        # 随机通道dropout
        if self.channel_dropout_prob > 0:
            mask = torch.bernoulli(
                torch.ones(eeg_sample.shape[0]) * (1 - self.channel_dropout_prob)
            )
            eeg_sample = eeg_sample * mask.unsqueeze(1)

        # 时间掩码（SpecAugment风格）
        if self.time_mask_prob > 0 and self.time_mask_size > 0:
            if torch.rand(1).item() < self.time_mask_prob:
                t = eeg_sample.size(1)
                start = torch.randint(0, max(1, t - self.time_mask_size), (1,)).item()
                eeg_sample[:, start : start + self.time_mask_size] = 0

        return eeg_sample


# ─── 训练器 ──────────────────────────────────────────────────────────────────


class EEGStressTrainer:
    """EEG压力/情绪识别模型训练器

    支持:
      - 回归（连续压力值）和分类（情绪标签）
      - 早停、学习率调度、梯度裁剪（论文[3]）
      - TensorBoard日志
      - K折交叉验证
      - 多模态训练（时域+频域）
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        config=None,
        task_type="regression",
    ):
        """
        Args:
            model:        EEG模型
            train_loader: 训练数据加载器
            val_loader:   验证数据加载器
            test_loader:  测试数据加载器
            config:       训练配置
            task_type:    'regression' | 'classification'
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_type = task_type

        default_config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 100,
            "warmup_epochs": 5,
            "label_smoothing": 0.0,
            "class_weight": "none",
            "mixed_precision": False,
            "seed": 42,
            "patience": 15,
            "min_delta": 1e-4,
            "lr_scheduler": "cosine",
            "gradient_clip": 1.0,
            "log_dir": "./logs",
            "model_save_dir": "./saved_models",
        }
        if config:
            default_config.update(config)
        self.config = default_config

        # 随机种子
        if self.config["seed"] is not None:
            seed = self.config["seed"]
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.device = torch.device(self.config["device"])
        self.model.to(self.device)

        # ── 损失函数 ──
        if task_type == "classification":
            ls = self.config["label_smoothing"]
            if self.config["class_weight"] == "balanced":
                # 从训练集计算权重
                all_labels = []
                for _, t in train_loader:
                    all_labels.append(t if not isinstance(t, tuple) else t[0])
                all_labels = torch.cat(all_labels)
                cls_counts = torch.bincount(all_labels).float()
                cls_weight = cls_counts.max() / cls_counts
                cls_weight[~torch.isfinite(cls_weight)] = 1.0  # empty classes → weight=1避免inf
                cls_weight = cls_weight.to(self.device)
            else:
                cls_weight = None
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=ls, weight=cls_weight
            )
        else:
            self.criterion = nn.MSELoss()

        # ── 混合精度 ──
        self.scaler = torch.amp.GradScaler() if self.config["mixed_precision"] else None

        # ── 优化器（论文[3]: Adam, lr=0.001）──
        opt_name = self.config["optimizer"]
        lr = self.config["learning_rate"]
        wd = self.config["weight_decay"]
        if opt_name == "adamw":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"未知优化器: {opt_name}，可选 adam/adamw/sgd")

        # 学习率调度
        if self.config["lr_scheduler"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config["epochs"]
            )
        elif self.config["lr_scheduler"] == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

        # 早停
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stop = False

        os.makedirs(self.config["log_dir"], exist_ok=True)
        os.makedirs(self.config["model_save_dir"], exist_ok=True)

        try:
            self.writer = SummaryWriter(log_dir=self.config["log_dir"])
        except Exception:
            # Windows中文字符路径会导致TensorFlow C++后端失败
            import tempfile
            safe_dir = os.path.join(tempfile.gettempdir(), "eeg_stress_tb")
            os.makedirs(safe_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=safe_dir)
            self.logger.warning(f"TensorBoard路径包含中文，已迁移到: {safe_dir}")
        self._setup_logging()

        # ── Visdom 实时可视化 ──
        self.visdom = None
        if _VISDOM_AVAILABLE and self.config.get("visdom_enabled", False):
            try:
                self.visdom = Visdom(
                    server=self.config.get("visdom_server", "http://localhost"),
                    port=self.config.get("visdom_port", 8097),
                    env=self.config.get("visdom_env", "eeg_stress"),
                    use_incoming_socket=False,
                )
                if self.visdom and self.visdom.check_connection():
                    self.visdom_windows = {}
                    self.logger.info(
                        f"Visdom 已连接: http://localhost:{self.config.get('visdom_port')}"
                    )
                else:
                    self.visdom = None
                    self.logger.info("Visdom 未连接 (server未启动)，训练继续")
            except Exception as e:
                self.visdom = None
                self.logger.info(f"Visdom 未连接，训练继续")

        self.train_losses_history = []
        self.val_losses_history = []
        self.metrics_history = []

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.config["log_dir"], "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> float:
        """训练一个epoch (支持混合精度)"""
        self.model.train()
        total_loss = 0.0

        for data, target in self.train_loader:
            if isinstance(target, tuple):
                target = target[0]
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                # 混合精度训练
                with torch.amp.autocast(self.device.type):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                if self.config["gradient_clip"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if self.config["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config["gradient_clip"]
                    )
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, data_loader) -> Tuple[float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for data, target in data_loader:
            if isinstance(target, tuple):
                target = target[0]
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)

            if self.task_type == "classification":
                loss = self.criterion(output, target)
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
            else:
                loss = self.criterion(output, target)
                all_preds.extend(output.cpu().numpy())

            total_loss += loss.item()
            all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        if self.task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(all_targets, all_preds),
                "loss": avg_loss,
            }
        else:
            metrics = {
                "mse": mean_squared_error(all_targets, all_preds),
                "mae": mean_absolute_error(all_targets, all_preds),
                "r2": r2_score(all_targets, all_preds),
                "correlation": float(np.corrcoef(all_targets, all_preds)[0, 1])
                if len(all_targets) > 1
                else 0.0,
            }

        return avg_loss, metrics, (all_targets, all_preds)

    def _update_visdom(self, epoch):
        """向 Visdom 推送实时训练指标"""
        if self.visdom is None:
            return

        # Loss 曲线
        if "loss" not in self.visdom_windows:
            self.visdom_windows["loss"] = self.visdom.line(
                X=np.array([0]), Y=np.array([self.train_losses_history[0]]),
                name="train_loss",
                opts=dict(title="训练/验证 Loss", xlabel="Epoch", ylabel="Loss"),
            )
            self.visdom.line(
                X=np.array([0]), Y=np.array([self.val_losses_history[0]]),
                win=self.visdom_windows["loss"], name="val_loss",
                update="new",
            )
        else:
            self.visdom.line(
                X=np.array([epoch]), Y=np.array([self.train_losses_history[-1]]),
                win=self.visdom_windows["loss"], name="train_loss",
                update="append",
            )
            self.visdom.line(
                X=np.array([epoch]), Y=np.array([self.val_losses_history[-1]]),
                win=self.visdom_windows["loss"], name="val_loss",
                update="append",
            )

        # 指标曲线（accuracy / mse / mae / r2 等）
        for key, val in self.metrics_history[-1].items():
            if key == "loss":
                continue
            win_key = f"metric_{key}"
            if win_key not in self.visdom_windows:
                self.visdom_windows[win_key] = self.visdom.line(
                    X=np.array([epoch]), Y=np.array([val]),
                    name=key,
                    opts=dict(title=f"指标: {key}", xlabel="Epoch", ylabel=key),
                )
            else:
                self.visdom.line(
                    X=np.array([epoch]), Y=np.array([val]),
                    win=self.visdom_windows[win_key], name=key,
                    update="append",
                )

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        torch.save(checkpoint, os.path.join(self.config["model_save_dir"], "latest_checkpoint.pth"))

        if is_best:
            torch.save(checkpoint, os.path.join(self.config["model_save_dir"], "best_model.pth"))
            self.logger.info(f"保存最佳模型 (val_loss={self.best_val_loss:.6f})")

    def train(self):
        """训练模型"""
        self.logger.info(f"开始训练 (task={self.task_type}, device={self.device})")
        self.logger.info(f"训练样本: {len(self.train_loader.dataset)}")
        self.logger.info(f"验证样本: {len(self.val_loader.dataset)}")

        train_losses, val_losses = [], []
        warmup_epochs = self.config.get("warmup_epochs", 0)
        base_lr = self.config["learning_rate"]

        for epoch in range(self.config["epochs"]):
            if self.early_stop:
                break

            # ── 学习率预热 ──
            if warmup_epochs > 0 and epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

            train_loss = self.train_epoch()
            val_loss, metrics, (y_true, y_pred) = self.validate(self.val_loader)

            # 更新学习率（仅在warmup结束后）
            if warmup_epochs == 0 or epoch >= warmup_epochs:
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # 日志
            metric_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items()
            )
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"train: {train_loss:.6f} | val: {val_loss:.6f} | {metric_str}"
            )

            # TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            for k, v in metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)

            # Visdom 实时更新
            self.train_losses_history.append(train_loss)
            self.val_losses_history.append(val_loss)
            self.metrics_history.append(metrics)
            self._update_visdom(epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 早停检查
            if val_loss < self.best_val_loss - self.config["min_delta"]:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=False)

            if self.patience_counter >= self.config["patience"]:
                self.logger.info(f"早停触发! {self.config['patience']}个epoch无改善")
                self.early_stop = True

        self._plot_loss_curve(train_losses, val_losses)

        if self.test_loader:
            self.test_model()

        self.writer.close()
        return train_losses, val_losses

    @torch.no_grad()
    def test_model(self):
        """测试模型"""
        self.logger.info("测试模型...")

        best_path = os.path.join(self.config["model_save_dir"], "best_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"加载最佳模型 (epoch={checkpoint['epoch']+1})")

        test_loss, metrics, (y_true, y_pred) = self.validate(self.test_loader)

        self.logger.info("测试结果:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.6f}")

        if self.task_type == "classification":
            self._plot_confusion_matrix(y_true, y_pred)
            self._plot_classification_report(y_true, y_pred)
        else:
            self._plot_predictions(y_true, y_pred)

    def _plot_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="训练损失", marker="o")
        plt.plot(val_losses, label="验证损失", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("训练和验证损失曲线")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config["log_dir"], "loss_curve.png"), dpi=150)
        plt.close()

    def _plot_predictions(self, y_true, y_pred):
        """绘制回归预测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 预测 vs 真实
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0, 0].plot(lims, lims, "r--", lw=2)
        axes[0, 0].set_xlabel("真实值")
        axes[0, 0].set_ylabel("预测值")
        axes[0, 0].set_title("预测 vs 真实值")
        axes[0, 0].grid(True, alpha=0.3)

        # 残差图
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("预测值")
        axes[0, 1].set_ylabel("残差")
        axes[0, 1].set_title("残差图")
        axes[0, 1].grid(True, alpha=0.3)

        # 误差分布
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("预测误差")
        axes[1, 0].set_ylabel("频数")
        axes[1, 0].set_title("误差分布")

        # 误差累积分布
        sorted_errors = np.sort(np.abs(residuals))
        cumsum = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumsum)
        axes[1, 1].set_xlabel("绝对误差")
        axes[1, 1].set_ylabel("累积比例")
        axes[1, 1].set_title("误差累积分布")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config["log_dir"], "prediction_analysis.png"), dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("混淆矩阵")
        plt.colorbar()

        classes = np.unique(np.concatenate([y_true, y_pred]))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel("真实标签")
        plt.xlabel("预测标签")
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["log_dir"], "confusion_matrix.png"), dpi=150)
        plt.close()

        # 打印分类报告
        report = classification_report(y_true, y_pred, digits=4)
        self.logger.info(f"分类报告:\n{report}")

        # 保存到文件
        with open(os.path.join(self.config["log_dir"], "classification_report.txt"), "w") as f:
            f.write(report)

    def _plot_classification_report(self, y_true, y_pred):
        """绘制分类指标条形图"""
        report = classification_report(y_true, y_pred, output_dict=True)
        classes = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]

        precision = [report[c]["precision"] for c in classes]
        recall = [report[c]["recall"] for c in classes]
        f1 = [report[c]["f1-score"] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, precision, width, label="Precision")
        plt.bar(x, recall, width, label="Recall")
        plt.bar(x + width, f1, width, label="F1-score")
        plt.xlabel("类别")
        plt.ylabel("分数")
        plt.title("分类指标")
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config["log_dir"], "classification_metrics.png"), dpi=150)
        plt.close()


# ─── 交叉验证 ────────────────────────────────────────────────────────────────


def cross_validate(
    model_factory,
    eeg_data,
    labels,
    n_folds=5,
    batch_size=32,
    task_type="regression",
    config=None,
    seed=42,
):
    """K折交叉验证（论文[3]）

    Args:
        model_factory:  返回模型实例的可调用对象
        eeg_data:       [n_samples, n_channels, n_timepoints]
        labels:         [n_samples]
        n_folds:        折数
        batch_size:     批次大小
        task_type:      'regression' | 'classification'
        config:         训练配置
        seed:           随机种子

    Returns:
        fold_results:  每折的评估指标
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(eeg_data)):
        print(f"\n{'='*50}")
        print(f"折 {fold + 1}/{n_folds}")
        print(f"{'='*50}")

        # 子集划分
        train_subset = Subset(EEGStressDataset(eeg_data, labels, task_type=task_type), train_idx)
        val_subset = Subset(EEGStressDataset(eeg_data, labels, task_type=task_type), val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = model_factory()

        # 配置
        fold_config = dict(config or {})
        fold_config["log_dir"] = f"./logs/fold_{fold + 1}"
        fold_config["model_save_dir"] = f"./saved_models/fold_{fold + 1}"

        # 训练
        trainer = EEGStressTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=fold_config,
            task_type=task_type,
        )
        trainer.train()

        # 最终验证
        _, metrics, _ = trainer.validate(val_loader)
        fold_results.append(metrics)
        print(f"折 {fold + 1} 结果: {metrics}")

    # 汇总
    print(f"\n{'='*50}")
    print(f"交叉验证结果 ({n_folds}折)")
    print(f"{'='*50}")

    avg_metrics = {}
    for key in fold_results[0].keys():
        values = [r[key] for r in fold_results]
        avg_metrics[key] = {"mean": np.mean(values), "std": np.std(values)}
        print(f"{key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return fold_results, avg_metrics


# ─── 主测试 ──────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    from model import EEGStressCNNLSTM

    print("=" * 60)
    print("EEG训练器测试")
    print("=" * 60)

    # 测试 回归任务
    print("\n[测试1] 回归 — 连续压力值预测")
    eeg_data, stress_labels = create_dummy_data(n_samples=200, task_type="regression")

    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        eeg_data, stress_labels, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    train_dataset = EEGStressDataset(X_train, y_train, task_type="regression",
                                     transform=EEGDataAugmentation(noise_std=0.01))
    val_dataset = EEGStressDataset(X_val, y_val, task_type="regression")
    test_dataset = EEGStressDataset(X_test, y_test, task_type="regression")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = EEGStressCNNLSTM(task_type="regression")

    trainer = EEGStressTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={
            "epochs": 10,
            "learning_rate": 1e-3,
            "log_dir": "./test_logs",
            "model_save_dir": "./test_models",
        },
        task_type="regression",
    )
    trainer.train()
    print("回归测试通过!")

    # 测试 分类任务
    print("\n[测试2] 分类 — 情绪类别识别")
    eeg_data_cls, labels_cls = create_dummy_data(n_samples=200, task_type="classification", n_classes=3)
    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
        eeg_data_cls, labels_cls, test_size=0.2, random_state=42
    )

    train_dataset_c = EEGStressDataset(X_train_c, y_train_c, task_type="classification")
    val_dataset_c = EEGStressDataset(X_val_c, y_val_c, task_type="classification")

    train_loader_c = DataLoader(train_dataset_c, batch_size=16, shuffle=True)
    val_loader_c = DataLoader(val_dataset_c, batch_size=16, shuffle=False)

    model_c = EEGStressCNNLSTM(task_type="classification", num_classes=3)

    trainer_c = EEGStressTrainer(
        model=model_c,
        train_loader=train_loader_c,
        val_loader=val_loader_c,
        config={"epochs": 5, "log_dir": "./test_logs_cls", "model_save_dir": "./test_models_cls"},
        task_type="classification",
    )
    trainer_c.train()
    print("分类测试通过!")

    # 测试 DE特征数据集
    print("\n[测试3] 差分熵(DE)特征数据集")
    de_dataset = EEGStressDataset(
        X_train, y_train, task_type="regression", extract_de=True, fs=128
    )
    sample, (label, de_feat, band_de) = de_dataset[0]
    print(f"  EEG: {sample.shape}, DE: {de_feat.shape}, Band-DE: {band_de.shape}")

    print("\n所有测试通过!")
