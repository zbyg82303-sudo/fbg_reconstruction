import collections
import json
import os
import queue
import sys
import threading

# 标定文件路径.
# PyInstaller 单文件打包后, 模块的 __file__ 指向解包临时目录 (_MEIPASS, 每次启动都换、
# 退出即删), 标定文件写在那里会丢失; 冻结运行必须改用 exe 所在目录.
if getattr(sys, "frozen", False):
    _CALIB_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    _CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(_CALIB_DIR, "calibration.json")

# 计算参数
points = 200
F_input = {"value": 30}
mode = {"type": "manual"}
F_prev = 30

# 线程间队列
queue_data = queue.Queue(maxsize=1)
queue_result = queue.Queue(maxsize=1)

# 数据采集
logger = None

# 位移云图状态
cloud_fig = None
cloud_ax = None
cloud_surfs = {}
cloud_colorbar = None
cloud_annotations = []

# 误差分析状态
error_window = None
error_text_widget = None
error_info_text = ""

# 固定坐标轴
AXIS_XLIM = (0, 300)
AXIS_YLIM = (-28, 28)
AXIS_ZLIM = (-65, 5)
CLOUD_DMAX = 60
CLOUD_GAMMA = 0.4

# 颜色标尺
COLOR_TICKS = [0, 2, 5, 10, 15, 20, 30, 40, 50, 60]

# 应变片位置
GAUGE_POSITIONS = [60, 120, 180, 240, 300]

# 通道配置 (0-based 索引，运行时由登录界面设置)
strain_channel_indices = [0, 1, 2, 3]   # 4个应变通道在原始数据中的索引
voltage_channel_index = 4                # 电压通道在原始数据中的索引

# 电压/位移状态
voltage_history = collections.deque(maxlen=100)
current_displacement_mm = 0.0
voltage_offset = 0.0  # 清零偏移

# 刷新率
REFRESH_HZ = {"value": 120}

# 延迟监控
latency_history = collections.deque(maxlen=60)

# 标定状态
calibration_window = None
calibration_data = []        # list of (load, [s1, s2, s3, s4])
calibration_coeffs = None    # fitted coefficients per channel
calibration_active = False   # whether calibration correction is applied
calibration_factors = None   # list of (offset_a, scale) per channel
current_strain_ue = None     # latest strain after zero-offset correction (με)
current_strain_raw = None   # latest raw strain from hardware (με), for zero calibration

# 清零偏移 (με)
strain_offsets = None

# 主 tk 根窗口
root_tk = None

# ---- 实时固有频率分析 (FFT) ----
# 仪器采样频率 SAMPLE_RATE_HZ (Hz): DHDAS 数据包内的采样行按此频率连续排列, 固定值.
# 分析前做 10:1 抗混叠抽取 (DECIMATE): 先低通再每 10 点取 1, 等效分析采样率
# ANALYSIS_RATE_HZ = 1000 Hz (Nyquist 500 Hz). FFT 频率轴只由分析采样率换算:
# Δf = ANALYSIS_RATE_HZ / 实际送入点数. 采样率与 FFT 检出的固有频率 peak_freq_hz
# 是两回事, 严禁混用.
SAMPLE_RATE_HZ = 10000.0
DECIMATE = 10
ANALYSIS_RATE_HZ = SAMPLE_RATE_HZ / DECIMATE

modal_window = None                      # 模态分析窗口 (Tk Toplevel)
modal_active = False                     # 仅在窗口打开时为 True, 关闭后零开销
modal_buffer = collections.deque()       # (整包采样矩阵 (n_rows, n_ch), 到达时间) 环形缓冲
modal_samples = 0                        # 缓冲内总采样行数 (用于按行数裁剪)
modal_max_samples = 163840               # 缓冲上限 (行), 约 16.4s @ 10000 Hz;
                                         # 8192 点 @1000 Hz 需 81920 行原始数据
modal_lock = threading.Lock()            # 保护 modal_buffer 的追加/快照
modal_result = None                      # 最新分析结果 dict (模态线程写入, UI 读取)
modal_params = {
    "fft_size": 4096,        # FFT 点数 N (固定, 每次只用最新 N 个分析点):
                             # 1000 Hz 分析采样率下 Δf = 0.24 Hz, 窗口 4.10 s;
                             # 可选 2048/4096/8192
    "avg_frames": 4,         # 多帧平均谱的帧数 (固有频率链路; 当前帧峰不受影响)
    "update_hz": 2.0,        # 频谱显示/识别刷新率; 原始数据仍按 1000 Hz 分析
    "max_freq": 100.0,       # 横轴显示终点; 可在模态窗口手动修改
    "min_show_freq": 0.0,    # 横轴显示起点
    "peak_count": 3,         # 检测峰值数量
    "min_freq": 1.0,         # 峰值搜索下限, 避开直流与低频漂移
    "min_gap_hz": 2.0,       # 峰值之间最小频率间隔
    "prominence_pct": 10.0,  # 峰值最小突出度 (相对谱峰值的百分比)
    "stability_pct": 2.0,    # 频率稳定性判定阈值 (%)
    "low_freq_max": 0.0,     # 不再强制低频峰作为一阶候选, 避免加载趋势抢占主读数
    "response_min_freq": 0.5,     # 响应主频搜索下限; 保留约 1 Hz 的静止/慢响应
    "response_max_freq": 120.0,   # 响应主频搜索上限; 按压/加载实验通常看低中频段
    "response_highpass_hz": 0.2,  # 轻微高通 + 去趋势, 抑制慢漂移但保留约 1 Hz 响应
    "notch_50hz": True,           # 静止时常见 50 Hz 工频干扰, 主频识别默认排除
    "channel": 0,            # 分析通道索引 0..3 (应变通道)
    "revision": 0,           # 参数版本号, 变化时重置峰值跟踪
}


def save_calibration():
    """Persist calibration state to JSON file."""
    data = {
        "active": calibration_active,
        "data": calibration_data,
        "coeffs": calibration_coeffs,
        "factors": calibration_factors,
    }
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"保存标定文件失败: {e}")


def load_calibration():
    """Restore calibration state from JSON file. Returns True if loaded."""
    if not os.path.exists(CALIBRATION_FILE):
        print(f"未找到标定文件: {CALIBRATION_FILE} (按无标定运行)")
        return False
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        global calibration_active, calibration_data, calibration_coeffs, calibration_factors
        calibration_active = data.get("active", False)
        calibration_data = data.get("data", [])
        calibration_coeffs = data.get("coeffs", None)
        calibration_factors = data.get("factors", None)
        print(f"已加载标定文件: {CALIBRATION_FILE}  active={calibration_active}")
        return True
    except Exception as e:
        print(f"加载标定文件失败: {e}")
        return False
