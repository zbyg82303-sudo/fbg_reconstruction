import collections
import json
import os
import queue

# 标定文件路径
_CALIB_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_FILE = os.path.join(_CALIB_DIR, "calibration.json")

# 计算参数
points = 200
F_input = {"value": 30}
mode = {"type": "manual"}
F_prev = 30

# 线程间队列
queue_data = queue.Queue(maxsize=3)
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
