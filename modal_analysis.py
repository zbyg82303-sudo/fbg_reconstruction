"""实时固有频率分析 — 对采集到的应变时域信号做滑动窗 FFT.

数据来源: threads.data_thread 从 DHDAS 收到的**整包采样行** (仪器 10000 Hz 连续时域
序列), 经 push_packet() 追加到 state.modal_buffer 环形缓冲. 本模块不读 socket、不新建
采集线程、不改变通信协议与原有队列。不能用"每包最后一行"做 FFT —— 那样每包只剩 1 个点,
序列变成按包到达率抽稀的信号, 频率轴会被错误放大。

采样率与抽取: 仪器采样率保留 SAMPLE_RATE_HZ = 10000 Hz 不变; 分析前做 10:1 抗混叠抽取
(state.DECIMATE), 等效分析采样率 ANALYSIS_RATE_HZ = 1000 Hz (Nyquist 500 Hz), 频率轴
只由它换算: Δf = ANALYSIS_RATE_HZ / 送入 FFT 的点数 N。抽取是"先低通, 再每 10 点取 1"
的真实降采样, 不是对频率结果做乘除修正, 固有频率读数不受影响、分辨率提高 10 倍。
peak_freq_hz 是 FFT 检出的结构固有频率, 与采样率是两回事, 严禁混用。

结果中两个量也必须分开 (不能拿当前帧峰值当固有频率):
  * cur_peaks — 当前帧 (实时) 频谱峰值, 每帧可变, 只用于显示"当前频谱峰";
  * orders    — 稳定固有频率: 多帧平均谱 → 峰值跟踪 → 滑动平均 + 稳定性判定,
                与当前帧是否恰好检出该峰无关 (允许单帧跳动)。

FFT 数据处理流程 (对齐 Rust 参考实现, 固定 N 点, N = fft_size 不随数据进入改变):
    x_raw     = 最近 N*10 个原始采样点 (抽取前, 含滤波边沿保护)
    x         = 抗混叠低通后每 10 点取 1 (分析采样率 1000 Hz)
    window    = np.hamming(N)                     # 先加窗, 不直接对原始信号做 FFT
    windowed  = (x - mean(x)) * window            # 去直流后再乘窗
    spectrum  = np.fft.rfft(windowed)             # rFFT, 只保留正频率 (N/2+1 点)
    spectrum  = spectrum / (N * np.mean(window))  # 先归一化, 再取模
    amplitude = np.abs(spectrum)                  # 非 DC / 非 Nyquist 分量乘 2
    frequency = np.arange(len(amplitude)) * ANALYSIS_RATE_HZ / N
FFT 本身只负责得到幅频谱; 固有频率识别 (限制一阶搜索频段 → 候选峰值 → 稳定确认)
由 detect_peaks / ModeTracker 完成, 二者不得混为一谈。
"""

import time
from collections import deque

import numpy as np
from scipy.signal import butter, decimate, filtfilt, find_peaks, iirnotch, sosfiltfilt

import state

_DECIM_PAD = 512          # 抽取滤波边沿保护 (原始采样行数)


# ===============================
# 数据接入 (由 data_thread 调用)
# ===============================
def push_packet(block, ts):
    """追加一个数据包的全部采样行 (n_rows, n_ch). 窗口未打开时立即返回."""
    if not state.modal_active:
        return
    try:
        arr = np.asarray(block, dtype=np.float32)
    except Exception:
        return
    if arr.ndim != 2 or arr.shape[0] == 0:
        return
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    with state.modal_lock:
        state.modal_buffer.append((arr, ts))
        state.modal_samples += arr.shape[0]
        while state.modal_samples > state.modal_max_samples and len(state.modal_buffer) > 1:
            old, _ = state.modal_buffer.popleft()
            state.modal_samples -= old.shape[0]


def clear_buffer():
    """清空缓冲 (打开窗口时调用, 避免混入上次的旧数据)."""
    with state.modal_lock:
        state.modal_buffer.clear()
        state.modal_samples = 0


def _raw_needed(n):
    """取 n 个分析点所需的原始采样行数 (10:1 抽取 + 滤波边沿保护)."""
    return int(n) * int(state.DECIMATE) + _DECIM_PAD


def _decimate_window(x):
    """10:1 抗混叠抽取: 先低通 (Chebyshev I 8 阶, 截止 400 Hz) 再每 10 点取 1.

    等效分析采样率 state.ANALYSIS_RATE_HZ = 1000 Hz, 保留 0~500 Hz 的固有频率信息.
    这是真实降采样, 不是对频率轴做乘除修正。
    """
    return decimate(x, int(state.DECIMATE), ftype="iir", zero_phase=True, axis=0)


def _filter_for_response(x, params):
    """预处理用于主响应频率识别的信号: 去趋势 + 可选高通 + 50 Hz 陷波."""
    y = np.asarray(x, dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size < 16:
        return y - y.mean()

    t = np.arange(y.size, dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    y = y - (slope * t + intercept)

    fs = float(state.ANALYSIS_RATE_HZ)
    nyq = fs / 2.0

    hp_hz = float(params.get("response_highpass_hz", 0.0) or 0.0)
    if 0.0 < hp_hz < nyq * 0.95 and y.size > 64:
        try:
            sos = butter(3, hp_hz, btype="highpass", fs=fs, output="sos")
            y = sosfiltfilt(sos, y)
        except Exception:
            pass

    if bool(params.get("notch_50hz", True)) and y.size > 64:
        notch_hz = 50.0
        if notch_hz < nyq * 0.95:
            try:
                b, a = iirnotch(notch_hz, Q=30.0, fs=fs)
                y = filtfilt(b, a, y)
            except Exception:
                pass
    return y


def _snapshot(n):
    """取最近 n 行原始采样样本 (抽取在 analyze_once 中统一做)."""
    with state.modal_lock:
        blocks = list(state.modal_buffer)
        rows = state.modal_samples
    if not blocks:
        return {"x": None, "rows": 0, "t1": 0.0, "gap": False}

    picked = []
    need = n
    for blk, ts in reversed(blocks):
        picked.append((blk, ts))
        need -= blk.shape[0]
        if need <= 0:
            break
    picked.reverse()

    x = np.concatenate([b for b, _ in picked], axis=0)
    if x.shape[0] > n:
        x = x[-n:]
    n_ch = x.shape[1]
    x = x.reshape(-1, n_ch)   # 保证二维

    t_list = [ts for _, ts in picked]
    gap = False
    if len(t_list) >= 8:
        dt = np.diff(np.asarray(t_list, dtype=float))
        med = float(np.median(dt))
        gap = bool(med > 0 and dt.max() > 5.0 * med)

    return {"x": x, "rows": rows, "t1": blocks[-1][1], "gap": gap}


# ===============================
# 频谱计算
# ===============================
_WINDOW_CACHE = {}


def _hamming(n):
    """固定 N 点的 Hamming 窗, 只创建一次, 不随每次 FFT 重建."""
    w = _WINDOW_CACHE.get(n)
    if w is None:
        w = np.hamming(n)
        _WINDOW_CACHE[n] = w
    return w


def amplitude_spectrum(x, n):
    """固定 N 点单边幅值谱 (流程对齐 Rust 参考实现; x 已是分析采样率的信号).

    N = n 由 fft_size 给定且固定, 每次只取**最新 N 个分析点** (由 10:1 抽取得到);
    数据持续进入只滑动窗口, 不改变点数, 也不重建缓冲/不用不定长数据做 FFT。

        window       = np.hamming(N)
        windowed     = (x - mean(x)) * window
        spectrum     = np.fft.rfft(windowed)                # 只保留正频率
        spectrum     = spectrum / (N * window_coeff)        # window_coeff = mean(window)
        amplitude    = np.abs(spectrum), 非 DC / 非 Nyquist 分量乘 2
        frequency[k] = k * ANALYSIS_RATE_HZ / N

    返回 (frequency, amplitude, N). 本函数只得到幅频谱; 固有频率识别在
    detect_peaks / ModeTracker 中另行完成。
    """
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    N = int(n)
    x = x[-N:]                          # 最新 N 个分析点 (窗口长度固定, 不因数据进入改变)

    window = _hamming(N)
    windowed = (x - x.mean()) * window  # 去直流 (抑制 DC 泄漏), 再乘窗

    spectrum = np.fft.rfft(windowed, n=N)      # rFFT: 只保留正频率, N/2+1 个点
    window_coeff = float(np.mean(window))
    spectrum = spectrum / (N * window_coeff)   # 先归一化, 再取模

    amplitude = np.abs(spectrum)
    if N % 2 == 0:
        amplitude[1:-1] *= 2            # 非 DC / 非 Nyquist 分量乘 2
    else:
        amplitude[1:] *= 2

    frequency = np.arange(amplitude.size) * float(state.ANALYSIS_RATE_HZ) / N
    return frequency, amplitude, N


# ===============================
# 峰值检测
# ===============================
def detect_peaks(freq_axis_hz, amps, min_freq, min_gap_hz, prom_frac, k,
                 low_freq_max=0.0):
    """find_peaks: 排除直流/低频, 最小突出度 + 最小频率间隔; 取幅值最大的 k 个.

    额外保证: low_freq_max > 0 时, 低频区 (≤ low_freq_max) 内幅值最大的峰一定入选
    (一阶模态往往不是全谱幅值最大的峰, 不能只按幅值排序把它挤掉).

    返回按频率升序的 [(peak_freq_hz, amp), ...].
    peak_freq_hz 是 FFT 检出的结构振动频率 (模态频率), 不是 SAMPLE_RATE_HZ.
    """
    if freq_axis_hz.size < 5:
        return []
    df_hz = float(freq_axis_hz[1] - freq_axis_hz[0])
    start = int(np.searchsorted(freq_axis_hz, max(min_freq, 0.0)))
    start = min(start, freq_axis_hz.size - 1)
    sub = amps[start:]
    if sub.size < 5:
        return []

    peak_max = float(np.max(sub))
    if not np.isfinite(peak_max) or peak_max <= 0:
        return []

    distance = max(1, int(round(min_gap_hz / df_hz)))
    idx, _ = find_peaks(sub, prominence=prom_frac * peak_max, distance=distance)
    if idx.size == 0:
        return []

    top = np.argsort(sub[idx])[::-1][:max(1, k)]     # 位置是 idx 内的下标
    sel = {int(i) for i in idx[top]}                 # 换成 sub 内的下标
    if low_freq_max > 0:
        # 低频区 (≤ low_freq_max) 内幅值最大的已检出峰, 无条件保留 (一阶候选)
        low_mask = freq_axis_hz[idx + start] <= low_freq_max
        if np.any(low_mask):
            cand = idx[low_mask]
            sel.add(int(cand[int(np.argmax(sub[cand]))]))
    sel = np.sort(np.fromiter(sel, dtype=int))
    return [(float(freq_axis_hz[i + start]), float(amps[i + start])) for i in sel]


def dominant_peak(freq_axis_hz, amps, min_freq, max_freq, min_gap_hz, prom_frac,
                  reject_hz=50.0, reject_band_hz=1.0):
    """在响应频带内按幅值选主峰, 用于识别加载/按压造成的主响应频率."""
    if freq_axis_hz.size < 5:
        return None

    upper = float(max_freq or 0.0)
    if upper <= 0:
        upper = float(freq_axis_hz[-1])
    mask = (freq_axis_hz >= float(min_freq)) & (freq_axis_hz <= upper)
    if not np.any(mask):
        return None

    f_band = freq_axis_hz[mask]
    a_band = amps[mask]
    peak_max = float(np.max(a_band)) if a_band.size else 0.0
    if not np.isfinite(peak_max) or peak_max <= 0:
        return None

    df_hz = float(freq_axis_hz[1] - freq_axis_hz[0])
    distance = max(1, int(round(min_gap_hz / df_hz)))
    idx, _ = find_peaks(a_band, prominence=prom_frac * peak_max, distance=distance)
    if idx.size == 0:
        return None

    candidates = []
    for i in idx:
        f_hz = float(f_band[i])
        if reject_hz > 0 and abs(f_hz - reject_hz) <= reject_band_hz:
            continue
        candidates.append((f_hz, float(a_band[i])))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p[1])


# ===============================
# 多帧谱平均
# ===============================
class SpectrumAverager:
    """连续多帧幅值谱的滑动平均, 抑制逐帧跳变 (同一频率轴才能平均).

    只作用于"模态频率"链路 (跟踪器输入); "当前频谱峰"仍用单帧谱, 保持实时性。
    """

    def __init__(self, max_frames=4):
        self.frames = deque(maxlen=max(1, int(max_frames)))

    def reset(self):
        self.frames.clear()

    def push(self, amps, max_frames=4):
        """追加一帧幅值谱, 返回 (平均谱, 参与平均的帧数)."""
        cap = max(1, int(max_frames))
        arr = np.asarray(amps, dtype=float)
        if self.frames.maxlen != cap:
            self.frames = deque(list(self.frames)[-cap:], maxlen=cap)
        if self.frames and self.frames[-1].size != arr.size:
            self.frames.clear()          # FFT 点数变了, 旧谱作废
        self.frames.append(arr)
        return np.mean(np.asarray(self.frames), axis=0), len(self.frames)


# ===============================
# 峰值跟踪 + 稳定性判断
# ===============================
class ModeTracker:
    """按频率邻近把逐帧峰值关联成"阶次"轨迹, 滑动平均平滑并判断稳定性.

    对应需求中的"频率稳定性判断 / 峰值跟踪": 只有最近若干帧持续检测到
    相近频率时, 才认为该阶模态频率稳定; 模态频率取轨迹的滑动平均,
    不直接采用当前帧峰值。
    """

    def __init__(self, history=8, match_tol=0.08, max_miss=3):
        self.history = history          # 平滑窗口帧数
        self.match_tol = match_tol      # 相邻帧频率关联容差 (相对值)
        self.max_miss = max_miss        # 连续丢失帧数上限, 超过则丢弃轨迹
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, peaks):
        """peaks: 本帧 [(peak_freq_hz, amp), ...] (峰值频率升序)."""
        peaks = list(peaks)
        used = set()

        for tr in self.tracks:
            ref_hz = tr["hist"][-1][0]
            best, best_d = None, None
            for i, (peak_freq_hz, _) in enumerate(peaks):
                if i in used:
                    continue
                d = abs(peak_freq_hz - ref_hz) / max(ref_hz, 1e-9)
                if d <= self.match_tol and (best_d is None or d < best_d):
                    best, best_d = i, d
            if best is None:
                tr["miss"] += 1
            else:
                tr["hist"].append(peaks[best])
                tr["miss"] = 0
                used.add(best)

        for i, pk in enumerate(peaks):
            if i not in used:
                self.tracks.append({"hist": deque([pk], maxlen=self.history), "miss": 0})

        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        self.tracks.sort(key=lambda t: float(np.median([f for f, _ in t["hist"]])))

    def orders(self, count, tol_frac, min_frames=4):
        """按频率从低到高给出各阶候选模态频率与稳定性.

        peak_freq_hz = 轨迹最近若干帧的滑动平均 (多帧确认后的模态频率, 允许落在
        两个 FFT 频点之间), 不是当前帧的峰值; last_hz 才是当前帧该轨迹的读出值。
        仅当最近一帧仍检测到该峰值 (miss == 0), 且连续多帧频率波动小于阈值时,
        才判定为"稳定"; 本帧丢失的轨迹保留显示但标记为识别中.
        """
        out = []
        for tr in self.tracks[:max(1, count)]:
            hist_hz = np.asarray([f for f, _ in tr["hist"]], dtype=float)
            peak_freq_hz = float(np.mean(hist_hz))      # 滑动平均 = 模态频率
            spread_hz = float(hist_hz.max() - hist_hz.min())
            n = int(hist_hz.size)
            stable = (tr["miss"] == 0 and n >= min_frames
                      and peak_freq_hz > 0 and spread_hz / peak_freq_hz <= tol_frac)
            out.append({
                "peak_freq_hz": peak_freq_hz,
                "last_hz": float(hist_hz[-1]),
                "amp": float(tr["hist"][-1][1]),
                "stable": bool(stable),
                "n": n,
                "miss": int(tr["miss"]),
                "spread_hz": spread_hz,
            })
        return out


class DominantFrequencyTracker:
    """跟踪主响应峰, 不按频率阶次排序, 而是平滑同一个主峰读数."""

    def __init__(self, history=6, max_jump_frac=0.35):
        self.history = deque(maxlen=max(1, int(history)))
        self.max_jump_frac = float(max_jump_frac)

    def reset(self):
        self.history.clear()

    def update(self, peak):
        if peak is None:
            return self.result(missing=True)
        f_hz, amp = peak
        if self.history:
            ref = float(np.median([f for f, _ in self.history]))
            if ref > 0 and abs(f_hz - ref) / ref > self.max_jump_frac:
                self.history.clear()
        self.history.append((float(f_hz), float(amp)))
        return self.result(missing=False)

    def result(self, missing=False):
        if not self.history:
            return None
        hist_hz = np.asarray([f for f, _ in self.history], dtype=float)
        f_hz = float(np.mean(hist_hz))
        spread_hz = float(hist_hz.max() - hist_hz.min())
        return {
            "freq_hz": f_hz,
            "last_hz": float(hist_hz[-1]),
            "amp": float(self.history[-1][1]),
            "n": int(hist_hz.size),
            "spread_hz": spread_hz,
            "stable": (not missing and hist_hz.size >= 3
                       and f_hz > 0 and spread_hz / f_hz <= 0.08),
            "missing": bool(missing),
        }


# ===============================
# 单次分析 (便于测试)
# ===============================
def analyze_once(snap, params, tracker, averager, response_tracker=None):
    """对缓冲快照做一次完整分析, 返回 result dict (不修改 state).

    原始数据先 10:1 抗混叠抽取到分析采样率 (ANALYSIS_RATE_HZ = 1000 Hz), 再取最新
    fft_size 个分析点做 FFT (固定 N 点, 不随数据进入变化), Δf = ANALYSIS_RATE_HZ / N。
    幅值谱由 amplitude_spectrum 得到, 固有频率识别 (限制一阶搜索频段 → 候选峰值 →
    稳定确认) 在其上另行进行。两条链路分开:
      * cur_peaks — 当前帧幅值谱的峰值 (实时, 每帧可变, 仅供显示);
      * orders    — 稳定固有频率: 多帧平均谱 → 峰值跟踪 → 滑动平均 + 稳定性判定。
    """
    n_req = int(params["fft_size"])
    n_have = int(snap["rows"]) // int(state.DECIMATE)   # 等效分析点数
    result = {
        "status": "collecting",
        "n_have": n_have,
        "n_need": n_req,
        "fft_size": n_req,
        "sample_rate_hz": float(state.SAMPLE_RATE_HZ),      # 仪器原始采样率 (固定)
        "analysis_rate_hz": float(state.ANALYSIS_RATE_HZ),  # 抽取后分析采样率 (频率轴依据)
        "n_fed": 0,            # 送入 FFT 的分析点数 (固定 = fft_size, 最新 N 点)
        "df_hz": 0.0,          # 频率分辨率 = ANALYSIS_RATE_HZ / n_fed
        "freq_axis_hz": None,  # FFT 频率轴: k * ANALYSIS_RATE_HZ / N
        "amps": None,          # 当前帧幅值谱
        "amps_avg": None,      # 多帧平均幅值谱
        "n_avg": 0,            # 参与平均的帧数
        "cur_peaks": [],       # 当前帧峰值 [(peak_freq_hz, amp), ...]
        "avg_peaks": [],       # 平均谱峰值 (跟踪器输入)
        "first_cur_hz": None,  # 当前帧一阶候选峰读出值 (实时, 与稳定值对照)
        "response_peak": None,  # 当前/加载响应主频: 指定频带内幅值最大的有效峰
        "orders": [],
        "max_show": 0.0,
        "gaps": False,
        "t": time.time(),
    }

    x_all = snap["x"]
    if x_all is None:
        return result

    if n_have < n_req:
        result["status"] = "waiting" if (time.time() - snap["t1"] > 1.5) else "collecting"
        return result

    ch = int(params.get("channel", 0))
    n_ch = x_all.shape[1]
    ch = max(0, min(ch, n_ch - 1))
    x_dec = _decimate_window(np.asarray(x_all[:, ch], dtype=float))
    x_resp = _filter_for_response(x_dec, params)

    freq_axis_hz, amps, n_fed = amplitude_spectrum(x_resp, n_req)
    df_hz = float(state.ANALYSIS_RATE_HZ) / n_fed

    k = int(params.get("peak_count", 3))
    min_freq = float(params.get("min_freq", 0.0))
    min_gap = float(params.get("min_gap_hz", 0.0))
    prom = float(params.get("prominence_pct", 10.0)) / 100.0
    low_freq_max = float(params.get("low_freq_max", 0.0) or 0.0)
    response_min = float(params.get("response_min_freq", min_freq) or min_freq)
    response_max = float(params.get("response_max_freq", 0.0) or 0.0)
    reject_50 = 50.0 if bool(params.get("notch_50hz", True)) else 0.0

    # 当前帧峰值 (实时, 允许跳变) — 只用于显示"当前频谱峰"
    cur_peaks = detect_peaks(freq_axis_hz, amps, min_freq, min_gap, prom, k,
                             low_freq_max)
    low_band = [p for p in cur_peaks if low_freq_max > 0 and p[0] <= low_freq_max]
    first_cur = float(max(low_band or cur_peaks, key=lambda p: p[1])[0]) if cur_peaks else None
    response_cur = dominant_peak(freq_axis_hz, amps, response_min, response_max,
                                 min_gap, prom, reject_50, 1.0)

    # 稳定固有频率链路: 多帧平均谱 → 峰值跟踪 → 滑动平均 + 稳定性判定
    amps_avg, n_avg = averager.push(amps, params.get("avg_frames", 4))
    avg_peaks = detect_peaks(freq_axis_hz, amps_avg, min_freq, min_gap, prom, k,
                             low_freq_max)
    tracker.update(avg_peaks)
    orders = tracker.orders(k, float(params.get("stability_pct", 2.0)) / 100.0)
    response_avg = dominant_peak(freq_axis_hz, amps_avg, response_min, response_max,
                                 min_gap, prom, reject_50, 1.0)
    if response_tracker is None:
        response_peak = {
            "freq_hz": float(response_avg[0]),
            "last_hz": float(response_avg[0]),
            "amp": float(response_avg[1]),
            "n": 1,
            "spread_hz": 0.0,
            "stable": False,
            "missing": False,
        } if response_avg else None
    else:
        response_peak = response_tracker.update(response_avg)

    nyquist = float(state.ANALYSIS_RATE_HZ) / 2.0
    max_show = float(params.get("max_freq", 0.0) or 0.0)
    if max_show <= 0 or max_show > nyquist:
        max_show = nyquist

    result.update({
        "status": "running",
        "freq_axis_hz": freq_axis_hz,
        "amps": amps,
        "amps_avg": amps_avg,
        "n_avg": n_avg,
        "n_fed": n_fed,
        "df_hz": df_hz,
        "cur_peaks": cur_peaks,
        "avg_peaks": avg_peaks,
        "first_cur_hz": first_cur,
        "response_cur_hz": float(response_cur[0]) if response_cur else None,
        "response_peak": response_peak,
        "orders": orders,
        "max_show": max_show,
        "gaps": bool(snap["gap"]),
    })
    return result


def _patch_status(status):
    """保留上一次频谱, 只改状态 (数据暂停时避免重复统计造成"假稳定")."""
    res = state.modal_result
    if res is None:
        return
    new = dict(res)
    new["status"] = status
    state.modal_result = new


# ===============================
# 分析线程
# ===============================
def modal_thread():
    """独立线程: 按设定刷新率计算 FFT 并发布结果, 不阻塞采集/重建."""
    tracker = ModeTracker()
    response_tracker = DominantFrequencyTracker()
    averager = SpectrumAverager()
    last_ts = None
    revision = -1

    while True:
        try:
            if not state.modal_active:
                if tracker.tracks:
                    tracker.reset()
                response_tracker.reset()
                averager.reset()
                last_ts = None
                revision = -1
                state.modal_result = None
                time.sleep(0.05)
                continue

            params = dict(state.modal_params)
            if params.get("revision") != revision:
                revision = params.get("revision")
                tracker.reset()
                response_tracker.reset()
                averager.reset()
                last_ts = None

            n = int(params["fft_size"])
            snap = _snapshot(_raw_needed(n))

            if snap["x"] is not None:
                data_ts = snap["t1"]
                if last_ts is not None and data_ts <= last_ts:
                    # 无新数据: 保留频谱, 不喂跟踪器
                    if state.modal_result is None or state.modal_result.get("status") != "waiting":
                        _patch_status("waiting")
                    time.sleep(0.05)
                    continue
                last_ts = data_ts

            state.modal_result = analyze_once(snap, params, tracker, averager, response_tracker)

            update_hz = float(params.get("update_hz", 5.0) or 5.0)
            time.sleep(max(0.02, min(1.0, 1.0 / max(update_hz, 0.5))))

        except Exception as e:
            print("模态分析线程错误:", e)
            time.sleep(0.5)
