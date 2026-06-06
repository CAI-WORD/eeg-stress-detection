#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 解决 OpenMP 运行时冲突 (PyTorch + MKL)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
EEG实时压力检测 — 连接W8采集数据 → 预处理 → 模型推理 → 压力趋势显示

数据流:
  W8 (500Hz, 8ch) → 重参考(TP9+TP10)/2 → 6信号通道
  → 带通0.5-45Hz → 降采样125Hz → 通道映射 → z-score归一化
  → 2.5s滑窗 → 模型推理 → 压力等级(0=放松, 1=中等, 2=高压)
"""

import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, lfilter
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QStatusBar,
    QVBoxLayout, QHBoxLayout, QProgressBar, QFrame,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

# 确保可以从项目根目录或eeg_stress_detection子目录导入model模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from eConEXG import iRecorder


# ─── 通道映射 ─────────────────────────────────────────────────────────────────
# 模型训练时使用的STEW通道顺序: FC5, P7, O1, O2, P8, FC6
# W8 6信号通道顺序: FCz(ch0), Pz(ch2), POz(ch3), O1(ch4), Oz(ch5), O2(ch6)
W8_TO_MODEL_CHANNEL_MAP = [0, 1, 2, 3, 4, 5]

# 压力等级名称
STRESS_NAMES = {0: "放松", 1: "中等压力", 2: "高压"}
STRESS_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}  # 绿/橙/红


class StressInferenceEngine:
    """EEG压力推理引擎"""

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: 模型权重路径 (None=用随机权重测试)
            device: 计算设备
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载模型（3分类：放松/中等压力/高压） ──
        from model import EEGStressCNNLSTM
        self.model = EEGStressCNNLSTM(
            n_channels=6,
            n_timepoints=320,
            cnn_channels=(32, 64, 128),
            cnn_kernel_sizes=(7, 5, 5),
            lstm_hidden=64,
            lstm_layers=2,
            lstm_bidirectional=True,
            lstm_dropout=0.4,
            dropout_rate=0.4,
            attention_heads=4,
            activation="gelu",
            task_type="classification",
            num_classes=3,  # 3分类：低/中/高压力
            use_2d_mapping=False,
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[引擎] 加载模型: {model_path}")
        else:
            print(f"[引擎] 警告: 模型文件不存在 '{model_path}'，使用随机权重")

        self.model.eval()

        # ── 滤波器参数 ──
        self.fs_in = 500       # W8采样率
        self.fs_out = 125      # 降采样目标
        self.dec_factor = 4    # 500/4=125Hz

        # 带通滤波器 0.5-45Hz
        self.sos_bandpass = butter(4, [0.5, 45], btype="band", fs=self.fs_out, output="sos")

        # ── 窗口参数 ──
        self.window_sec = 2.5
        self.window_len = int(self.fs_out * self.window_sec)  # 312点

        # ── 滑动缓冲区 ──
        self.buffer = deque(maxlen=self.window_len * self.dec_factor * 2)

        # ── 平滑输出 ──
        self.smooth_window = 3
        self.stress_history = deque(maxlen=self.smooth_window)

        # ── 计数器 ──
        self.samples_collected = 0
        self.last_infer_time = 0.0
        self.infer_interval = 0.5

    def re_reference(self, raw_8ch):
        """重参考: (TP9 + TP10) / 2

        Args:
            raw_8ch: [8] numpy数组, W8原始8通道数据

        Returns:
            [6] 重参考后的6信号通道 [FCz, Pz, POz, O1, Oz, O2]
        """
        tp9 = raw_8ch[1]
        tp10 = raw_8ch[7]
        ref = (tp9 + tp10) / 2.0

        signal_chs = [raw_8ch[0], raw_8ch[2], raw_8ch[3], raw_8ch[4], raw_8ch[5], raw_8ch[6]]
        referenced = np.array([ch - ref for ch in signal_chs], dtype=np.float64)
        return referenced

    def channel_mapping(self, w8_6ch):
        """将W8 6通道映射到模型输入顺序

        W8顺序: FCz(0), Pz(1), POz(2), O1(3), Oz(4), O2(5)
        模型输入: FC5(0), P7(1), O1(2), O2(3), P8(4), FC6(5)
        映射: [FCz→FC5, Pz→P7, POz→P8, O1→O1, Oz→FC6, O2→O2]
        """
        mapping = [0, 1, 4, 3, 5, 2]
        if w8_6ch.ndim == 1:
            return w8_6ch[mapping]
        return w8_6ch[mapping, :]

    def preprocess(self, raw_8ch_frame):
        """预处理单帧W8数据"""
        ref_6ch = self.re_reference(raw_8ch_frame)
        self.samples_collected += 1
        return ref_6ch

    def downsample_and_process(self):
        """缓冲区数据 → 降采样 → 通道映射 → 滤波 → 归一化

        Returns:
            [6, T] 处理后的数据, 或 None
        """
        buf = np.array(self.buffer)
        n_samples = buf.shape[0]
        need_raw = self.window_len * self.dec_factor  # 1248点
        if n_samples < need_raw:
            return None

        recent = buf[-need_raw:, :]
        downsampled = recent[::self.dec_factor, :].T  # [6, 312]
        mapped = self.channel_mapping(downsampled)

        filtered = np.zeros_like(mapped)
        for ch in range(6):
            filtered[ch] = sosfilt(self.sos_bandpass, mapped[ch])

        normalized = np.zeros_like(filtered)
        for ch in range(6):
            ch_mean = filtered[ch].mean()
            ch_std = filtered[ch].std() + 1e-8
            normalized[ch] = (filtered[ch] - ch_mean) / ch_std
            normalized[ch] = np.clip(normalized[ch], -5, 5)

        return normalized  # [6, 312]

    @torch.no_grad()
    def infer(self, eeg_window):
        """模型推理

        Args:
            eeg_window: [6, 312] numpy数组

        Returns:
            stress_level: 压力等级 (0=放松, 1=中等, 2=高压)
            probabilities: [p_low, p_med, p_high]
        """
        tensor = torch.FloatTensor(eeg_window).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        self.stress_history.append(pred)
        smoothed = int(round(np.mean(self.stress_history)))

        return smoothed, probs.cpu().numpy()[0]

    def feed_frame(self, raw_8ch_frame):
        """输入一帧W8原始数据, 如果窗口满则返回推理结果

        Args:
            raw_8ch_frame: [8] numpy数组

        Returns:
            dict or None:
                {"stress_level": int(0-2), "probabilities": [3], "signal": [6,T]}
        """
        now = time.time()

        ref_6ch = self.preprocess(raw_8ch_frame)
        self.buffer.append(ref_6ch)

        if now - self.last_infer_time < self.infer_interval:
            return None

        processed = self.downsample_and_process()
        if processed is None:
            return None

        stress_level, probs = self.infer(processed)
        self.last_infer_time = now

        return {
            "stress_level": int(stress_level),
            "probabilities": [float(p) for p in probs],
            "signal": processed,
        }

    def reset(self):
        """重置缓冲区"""
        self.buffer.clear()
        self.stress_history.clear()
        self.samples_collected = 0


# ─── 实时压力检测GUI ──────────────────────────────────────────────────────────

class StressDetectionGUI(QWidget):
    """实时压力检测GUI — EEG信号波形 + 压力等级 + 趋势图"""

    STRESS_NAMES = {0: "放松", 1: "中等压力", 2: "高压"}
    STRESS_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
    CHANNEL_NAMES = ["FCz", "Pz", "POz", "O1", "Oz", "O2"]

    def __init__(self):
        super().__init__()
        self.selected_channel_index = 0
        self.current_stress_level = 0
        self.current_probs = [1.0, 0.0, 0.0]

        self.init_device()
        self.init_engine()
        self.init_display_buffers()
        self.init_stress_history()
        self.set_layout()
        self.setup_timer()

    # ── 设备连接 ──────────────────────────────────────────────────────────────

    def init_device(self):
        """连接W8脑电放大器"""
        self.dev = iRecorder(dev_type="USB8")
        self.dev.set_frequency(500)
        self.dev.find_devs()
        while True:
            ret = self.dev.get_devs()
            if ret:
                break
        self.dev.connect_device(ret[0])
        self.is_recording = False

    # ── 推理引擎 ──────────────────────────────────────────────────────────────

    def init_engine(self):
        """初始化压力推理引擎"""
        model_path = os.path.join(_SCRIPT_DIR, "stew_models_6ch", "best_model.pth")
        self.engine = StressInferenceEngine(model_path=model_path)

    # ── 显示缓冲区与滤波器（同test1.py） ──────────────────────────────────────

    def init_display_buffers(self):
        """初始化用于信号显示的滤波器和缓冲区"""
        fs = 500
        self.buffer_N = 10000            # 缓冲区大小 20s
        self.display_N = 2000            # 显示窗口 4s
        self.freq_display = np.fft.fftfreq(self.buffer_N, 1 / fs)

        # 显示用滤波器：带通2-150Hz + 陷波48-52Hz + 陷波98-102Hz
        self.b1, self.a1 = butter(2, [2, 150], fs=500, btype='band', analog=False)
        self.b2, self.a2 = butter(2, [48, 52], fs=500, btype='bandstop', analog=False)
        self.b3, self.a3 = butter(2, [98, 102], fs=500, btype='bandstop', analog=False)

        self.z1 = np.zeros((6, max(len(self.b1), len(self.a1)) - 1))
        self.z2 = np.zeros((6, max(len(self.b2), len(self.a2)) - 1))
        self.z3 = np.zeros((6, max(len(self.b3), len(self.a3)) - 1))
        self.signal_raw = np.zeros((8, self.buffer_N))
        self.signal_reref = np.zeros((6, self.buffer_N))
        self.signal_filtered = np.zeros((6, self.buffer_N))

        # 时间轴
        self.display_time = [k / 500 for k in range(self.display_N)]

    # ── 压力历史（趋势图用） ──────────────────────────────────────────────────

    def init_stress_history(self):
        """压力历史记录"""
        self.stress_levels = []     # 压力等级序列 [0,1,2,...]
        self.stress_times = []      # 时间戳
        self.stress_confs = []      # 对应置信度
        self.max_history = 80       # 保留最近80次推理

    # ── 定时器 ────────────────────────────────────────────────────────────────

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.setInterval(100)  # 100ms更新

    # ── GUI布局 ──────────────────────────────────────────────────────────────

    def set_layout(self):
        self.setWindowTitle('EEG实时压力检测')
        self.resize(900, 750)

        # ── 图1: EEG信号 ──
        self.figure1 = Figure(figsize=(9, 2.8))
        self.canvas1 = FigureCanvas(self.figure1)
        self.axes1 = self.figure1.add_subplot(111)
        self.line1, = self.axes1.plot([], [], color='#2196F3', linewidth=0.8)
        self.axes1.set_xlim(0, 4)
        self.axes1.set_ylim(-5, 5)
        self.axes1.set_ylabel('Amplitude (µV)')
        self.axes1.grid(True, alpha=0.3)
        self.figure1.tight_layout(pad=1.5)

        # ── 压力指示器（自定义样式QLabel） ──
        self.stress_label = QLabel("放松")
        self.stress_label.setAlignment(Qt.AlignCenter)
        self.stress_label.setFont(QFont("Microsoft YaHei UI", 28, QFont.Bold))
        self.stress_label.setFixedHeight(70)
        self.stress_label.setStyleSheet(
            f"background-color: {STRESS_COLORS[0]}; color: white; "
            f"border-radius: 10px; padding: 5px;"
        )

        # 置信度进度条
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFixedHeight(20)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("置信度: %p%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 5px;
                text-align: center;
                background: #f0f0f0;
            }
            QProgressBar::chunk {
                background: #4CAF50;
                border-radius: 4px;
            }
        """)

        # 压力等级文字指示
        self.level_detail = QLabel("放松  |  压力值: 0/2")
        self.level_detail.setAlignment(Qt.AlignCenter)
        self.level_detail.setFont(QFont("Microsoft YaHei UI", 10))
        self.level_detail.setStyleSheet("color: #666; padding: 2px;")

        # ── 图2: 压力趋势图 ──
        self.figure2 = Figure(figsize=(9, 2.5))
        self.canvas2 = FigureCanvas(self.figure2)
        self.axes2 = self.figure2.add_subplot(111)
        self.line2, = self.axes2.plot([], [], color='#F44336', linewidth=1.5, drawstyle='steps-post')
        self.axes2.set_xlabel('Time (s)')
        self.axes2.set_ylabel('Stress Level')
        self.axes2.set_ylim(-0.2, 2.2)
        self.axes2.set_xlim(0, 40)
        self.axes2.set_yticks([0, 1, 2])
        self.axes2.set_yticklabels(['Low', 'Med', 'High'])
        self.axes2.grid(True, alpha=0.3)
        self.figure2.tight_layout(pad=1.5)

        # ── 按钮 ──
        self.btn_start = QPushButton("▶ 开始")
        self.btn_start.setFixedHeight(35)
        self.btn_start.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; padding: 5px 20px; "
            "background-color: #2196F3; color: white; border: none; border-radius: 5px; }"
            "QPushButton:hover { background-color: #1976D2; }"
        )
        self.btn_start.clicked.connect(self.btn_start_clicked)

        self.channel_buttons = []
        for i, name in enumerate(self.CHANNEL_NAMES):
            btn = QPushButton(name)
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                "QPushButton { padding: 3px 10px; border: 1px solid #ccc; "
                "border-radius: 4px; background: #f5f5f5; }"
                "QPushButton:hover { background: #e0e0e0; }"
            )
            btn.clicked.connect(lambda _, idx=i: self.show_single_channel(idx))
            self.channel_buttons.append(btn)

        # 默认高亮第一个通道
        self.channel_buttons[0].setStyleSheet(
            "QPushButton { padding: 3px 10px; border: 1px solid #4CAF50; "
            "border-radius: 4px; background: #4CAF50; color: white; }"
        )

        # ── 状态栏 ──
        self.statusBar = QStatusBar(self)
        self.statusBar.showMessage("就绪 — 点击「开始」连接设备")

        # ── 布局组装 ──
        main_layout = QVBoxLayout()
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 信号区
        signal_layout = QVBoxLayout()
        signal_layout.addWidget(QLabel("EEG 信号"))
        signal_layout.addWidget(self.canvas1)

        # 压力区
        stress_layout = QVBoxLayout()
        stress_layout.setSpacing(5)
        stress_layout.addWidget(self.stress_label)
        stress_layout.addWidget(self.confidence_bar)
        stress_layout.addWidget(self.level_detail)

        # 趋势区
        trend_layout = QVBoxLayout()
        trend_layout.addWidget(QLabel("压力趋势"))
        trend_layout.addWidget(self.canvas2)

        # 按钮区
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_start)
        button_layout.addSpacing(10)
        for btn in self.channel_buttons:
            button_layout.addWidget(btn)
        button_layout.addStretch()

        # 合并
        main_layout.addLayout(signal_layout)
        main_layout.addLayout(stress_layout)
        main_layout.addLayout(trend_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.statusBar)

        self.setLayout(main_layout)

    # ── 定时更新 ──────────────────────────────────────────────────────────────

    def update_display(self):
        """定时读取设备数据并更新所有显示"""
        if not self.is_recording:
            return

        try:
            frames = self.dev.get_data(timeout=0.02)
        except Exception as e:
            self.statusBar.showMessage(f"读取数据失败: {e}")
            return

        for frame in frames:
            # 更新显示缓冲区（滤波后用于波形显示）
            self.update_display_buffer(frame)

            # 送入推理引擎
            result = self.engine.feed_frame(frame)
            if result:
                self.on_stress_result(result)

        # 更新图表
        self.update_signal_plot()
        self.update_trend_plot()

    def update_display_buffer(self, frame):
        """处理单帧数据：重参考 + 滤波（同test1.py）"""
        # 更新原始数据缓冲区（滚动）
        for ch_idx in range(8):
            self.signal_raw[ch_idx] = np.roll(self.signal_raw[ch_idx], -1)
            self.signal_raw[ch_idx][-1] = frame[ch_idx]

        # 重参考
        ref = (self.signal_raw[1][-1] + self.signal_raw[7][-1]) / 2
        self.signal_reref[0][-1] = self.signal_raw[0][-1] - ref
        self.signal_reref[1][-1] = self.signal_raw[2][-1] - ref
        self.signal_reref[2][-1] = self.signal_raw[3][-1] - ref
        self.signal_reref[3][-1] = self.signal_raw[4][-1] - ref
        self.signal_reref[4][-1] = self.signal_raw[5][-1] - ref
        self.signal_reref[5][-1] = self.signal_raw[6][-1] - ref

        # 级联滤波
        for ch_idx in range(6):
            s1, self.z1[ch_idx] = lfilter(
                self.b1, self.a1, [self.signal_reref[ch_idx][-1]], zi=self.z1[ch_idx]
            )
            s2, self.z2[ch_idx] = lfilter(
                self.b2, self.a2, [s1[0]], zi=self.z2[ch_idx]
            )
            s3, self.z3[ch_idx] = lfilter(
                self.b3, self.a3, [s2[0]], zi=self.z3[ch_idx]
            )
            self.signal_filtered[ch_idx] = np.roll(self.signal_filtered[ch_idx], -1)
            self.signal_filtered[ch_idx][-1] = s3[0]

    def on_stress_result(self, result):
        """处理新的推理结果"""
        level = result["stress_level"]
        probs = result["probabilities"]

        self.current_stress_level = level
        self.current_probs = probs

        # 记录历史
        now = time.time()
        self.stress_levels.append(level)
        self.stress_times.append(now)
        self.stress_confs.append(probs[level])

        if len(self.stress_levels) > self.max_history:
            self.stress_levels.pop(0)
            self.stress_times.pop(0)
            self.stress_confs.pop(0)

        # 更新压力指示器
        self.update_stress_indicator()

    # ── 显示更新 ──────────────────────────────────────────────────────────────

    def update_signal_plot(self):
        """更新EEG信号波形图"""
        self.line1.set_data(
            self.display_time,
            self.signal_filtered[self.selected_channel_index][-self.display_N:]
        )
        self.axes1.set_xlim(self.display_time[0], self.display_time[-1])
        y_data = self.signal_filtered[self.selected_channel_index][-self.display_N:]
        if len(y_data) > 0 and y_data.max() != y_data.min():
            margin = max(abs(y_data).max() * 0.2, 1.0)
            self.axes1.set_ylim(y_data.min() - margin, y_data.max() + margin)
        self.canvas1.draw_idle()

    def update_stress_indicator(self):
        """更新压力等级指示灯"""
        level = self.current_stress_level
        conf = self.current_probs[level]
        name = self.STRESS_NAMES.get(level, "未知")
        color = self.STRESS_COLORS.get(level, "#999")

        self.stress_label.setText(f"{name}")
        self.stress_label.setStyleSheet(
            f"background-color: {color}; color: white; "
            f"border-radius: 10px; padding: 5px;"
        )

        self.confidence_bar.setValue(int(conf * 100))
        bar_color = color
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ddd; border-radius: 5px;
                text-align: center; background: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background: {bar_color}; border-radius: 4px;
            }}
        """)

        # 统计：各等级占比
        if self.stress_levels:
            recent = self.stress_levels[-20:]
            pct_low = recent.count(0) / len(recent) * 100
            pct_med = recent.count(1) / len(recent) * 100
            pct_high = recent.count(2) / len(recent) * 100
            self.level_detail.setText(
                f"放松 {pct_low:.0f}%  |  中等 {pct_med:.0f}%  |  高压 {pct_high:.0f}%  "
                f"|  当前 {name}  (置信度 {conf*100:.0f}%)"
            )

    def update_trend_plot(self):
        """更新压力趋势图"""
        if len(self.stress_levels) < 2:
            return

        # 相对时间（秒）
        t0 = self.stress_times[0]
        rel_times = [t - t0 for t in self.stress_times]

        self.line2.set_data(rel_times, self.stress_levels)
        self.axes2.set_xlim(rel_times[0], rel_times[-1] + 2)
        self.canvas2.draw_idle()

    # ── 按钮事件 ──────────────────────────────────────────────────────────────

    def btn_start_clicked(self):
        if self.is_recording:
            # 停止采集
            self.dev.stop_acquisition()
            self.is_recording = False
            self.timer.stop()
            self.btn_start.setText("▶ 开始")
            self.btn_start.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: bold; padding: 5px 20px; "
                "background-color: #2196F3; color: white; border: none; border-radius: 5px; }"
                "QPushButton:hover { background-color: #1976D2; }"
            )
            self.statusBar.showMessage("采集已停止")
        else:
            # 开始采集
            self.dev.start_acquisition_data()
            self.is_recording = True
            self.timer.start()
            self.btn_start.setText("■ 停止")
            self.btn_start.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: bold; padding: 5px 20px; "
                "background-color: #F44336; color: white; border: none; border-radius: 5px; }"
                "QPushButton:hover { background-color: #D32F2F; }"
            )
            self.statusBar.showMessage("采集中...")

    def show_single_channel(self, channel_index):
        """选择显示的通道"""
        self.selected_channel_index = channel_index
        for i, btn in enumerate(self.channel_buttons):
            if i == channel_index:
                btn.setStyleSheet(
                    "QPushButton { padding: 3px 10px; border: 1px solid #4CAF50; "
                    "border-radius: 4px; background: #4CAF50; color: white; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { padding: 3px 10px; border: 1px solid #ccc; "
                    "border-radius: 4px; background: #f5f5f5; }"
                    "QPushButton:hover { background: #e0e0e0; }"
                )
        self.update_signal_plot()

    def __del__(self):
        if hasattr(self, 'dev'):
            self.dev.close_dev()


# ─── 主入口 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = StressDetectionGUI()
    w.show()
    app.exec()
