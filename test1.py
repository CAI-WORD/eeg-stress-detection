from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QStatusBar
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QSizePolicy
import sys
from eConEXG import iRecorder
from PyQt5.QtCore import QTimer, Qt
import numpy as np
from scipy.signal import butter, lfilter


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.init_device()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plots)
        self.init_filters_and_buffers()
        self.set_layout()
        self.selected_channel_index = 0

    def init_device(self):
        self.dev = iRecorder(dev_type="USB8")
        self.dev.set_frequency(500)
        self.dev.find_devs()
        while True:
            ret = self.dev.get_devs()
            if ret:
                break
        self.dev.connect_device(ret[0])
        self.is_recording = False

    def init_filters_and_buffers(self):
        fs = 500 
        self.N = 10000 
        self.half_N = self.N // 2 + 1
        self.freqs = np.fft.fftfreq(self.N, 1 / fs)
            
        self.b1, self.a1 = butter(2, [2, 150], fs=500, btype='band', analog=False)        
        self.b2, self.a2 = butter(2, [48, 52], fs=500, btype='bandstop', analog=False)
        self.b3, self.a3 = butter(2, [98, 102], fs=500, btype='bandstop', analog=False)
        
        self.z1 = np.zeros((6, max(len(self.b1), len(self.a1)) - 1))
        self.z2 = np.zeros((6, max(len(self.b2), len(self.a2)) - 1))
        self.z3 = np.zeros((6, max(len(self.b3), len(self.a3)) - 1))
        self.signal_raw = np.zeros((8, 10000))
        self.signal_reref = np.zeros((6, 10000))
        self.signal_filtered = np.zeros((6, 10000))
        self.signal = [np.zeros(1000) for _ in range(6)]
        self.fft_result = [np.zeros(1000, dtype=complex) for _ in range(6)]

   
    def update_plots(self):
        if self.is_recording:
            frames = self.dev.get_data(timeout=0.02)
            for frame in frames:
                self.t.pop(0)
                self.t.append(self.t[-1] + 0.002)
                for ch_idx in range(8):                
                    self.signal_raw[ch_idx] = np.roll(self.signal_raw[ch_idx], -1)
                    self.signal_raw[ch_idx][-1] = frame[ch_idx]
                
                referenced_sample=(self.signal_raw[1][-1] + self.signal_raw[7][-1]) / 2
                self.signal_reref[0][-1]=self.signal_raw[0][-1]-referenced_sample
                self.signal_reref[1][-1]=self.signal_raw[2][-1]-referenced_sample
                self.signal_reref[2][-1]=self.signal_raw[3][-1]-referenced_sample
                self.signal_reref[3][-1]=self.signal_raw[4][-1]-referenced_sample
                self.signal_reref[4][-1]=self.signal_raw[5][-1]-referenced_sample
                self.signal_reref[5][-1]=self.signal_raw[6][-1]-referenced_sample
                
                for ch_idx in range(6):
                    filtered_sample1, self.z1[ch_idx] = lfilter(self.b1, self.a1, [self.signal_reref[ch_idx][-1]], zi=self.z1[ch_idx])
                    filtered_sample2, self.z2[ch_idx] = lfilter(self.b2, self.a2, [filtered_sample1[0]], zi=self.z2[ch_idx])
                    filtered_sample3, self.z3[ch_idx] = lfilter(self.b3, self.a3, [filtered_sample2[0]], zi=self.z3[ch_idx])
                    self.signal_filtered[ch_idx] = np.roll(self.signal_filtered[ch_idx], -1)
                    self.signal_filtered[ch_idx][-1] = filtered_sample3[0]
            self.update_signal_plot()

    def update_signal_plot(self):
        self.line1.set_data(self.t, self.signal_filtered[self.selected_channel_index])
        self.axes1.set_xlim([self.t[0], self.t[-1]])
        self.axes1.set_ylim([np.min(self.signal_filtered[self.selected_channel_index]), np.max(self.signal_filtered[self.selected_channel_index])])
        self.canvas1.draw_idle()
        
        fft_result = np.fft.fft(self.signal_filtered[self.selected_channel_index][-self.N:])
        fft_result = np.abs(fft_result[:self.half_N])
        self.line2.set_data(self.freqs[:self.half_N-1], fft_result[:-1])
        self.axes2.set_xlim([0, 8])
        self.axes2.relim()
        self.axes2.autoscale_view()
        self.canvas2.draw_idle()
            
    def set_layout(self):
        self.setWindowTitle('My Brain Wave')
        self.resize(800, 600)

        self.figure1 = Figure()
        self.figure2 = Figure()
        self.canvas1 = FigureCanvas(self.figure1)
        self.canvas2 = FigureCanvas(self.figure2)
              
        self.btn1 = QPushButton("start", self)
        self.btn1.clicked.connect(self.btn1_clicked)
        
        self.channel_buttons = [QPushButton(f"Channel {i + 1}", self) for i in range(6)]
        self.channel_buttons[0].setText('FCz')
        self.channel_buttons[1].setText('Pz')
        self.channel_buttons[2].setText('Poz')
        self.channel_buttons[3].setText('O1')
        self.channel_buttons[4].setText('Oz')
        self.channel_buttons[5].setText('O2')
        self.channel_buttons[0].setStyleSheet("QPushButton { background-color: green; }")
        

        for ch_idx, button in enumerate(self.channel_buttons):
            button.clicked.connect(lambda _, index=ch_idx: self.show_single_channel(index))

        self.statusBar = QStatusBar(self)
        self.statusBar.showMessage("Ready")

        main_layout = QVBoxLayout()
        canvas1_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        canvas1_layout.addWidget(self.canvas1)
        canvas1_layout.addWidget(self.canvas2)
        button_layout.addWidget(self.btn1)
        button_layout.addStretch(1)

        for button in self.channel_buttons:
            button_layout.addWidget(button)

        main_layout.addLayout(canvas1_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.statusBar)

        self.setLayout(main_layout)
        
        self.axes1 = self.figure1.add_subplot(111)
        self.axes2 = self.figure2.add_subplot(111)
        self.axes2.xaxis.set_minor_locator(MultipleLocator(1))
        self.t = [k / 500 for k in range(10000)]
        self.line1, = self.axes1.plot([], [])
        self.line2, = self.axes2.plot([], [])

        self.axes1.set_xlim([0, 10])
        self.axes1.set_xlabel('Time (s)')
        self.axes1.set_ylabel(r'Amplitude (\mu V) ')
        self.axes1.set_ylim([-0.5, 7.5])
        self.axes1.grid(True)

        self.axes2.set_xlabel('Frequency (Hz)')
        self.axes2.set_ylabel('Amplitude (dB)')
        self.axes2.grid(True)


    def btn1_clicked(self):
        if self.is_recording: 
            #stop_recording
            self.dev.stop_acquisition()
            self.is_recording = False
            self.timer.stop()
            self.btn1.setText("Continue")
            self.statusBar.showMessage("Recording stopped")
        else: 
            #start_recording            
            self.dev.start_acquisition_data()
            self.is_recording = True
            self.timer.start(100)
            self.btn1.setText("Stop")
            self.statusBar.showMessage("Start Recording...")
            
    def show_single_channel(self, channel_index):
        self.selected_channel_index = channel_index
        for button in self.channel_buttons:
            button.setStyleSheet("")
        self.channel_buttons[channel_index].setStyleSheet("QPushButton { background-color: green; }")
        self.update_signal_plot()

    def __del__(self):
        self.dev.close_dev()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    app.exec()
