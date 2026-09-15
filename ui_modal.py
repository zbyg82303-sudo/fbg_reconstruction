import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import modal_analysis
import state


# 应变通道 → 梁位置 (与 reconstruction.sensor_positions 对应)
CH_LABELS = ["通道1 (60mm)", "通道2 (120mm)", "通道3 (180mm)", "通道4 (240mm)"]
FFT_SIZES = ["2048", "4096", "8192"]

COLOR_OK = "#2E7D32"      # 稳定
COLOR_WARN = "#EF6C00"    # 识别中/不稳定
COLOR_IDLE = "#999999"


def open_modal_window(_event=None):
    if state.modal_window is not None:
        try:
            state.modal_window.lift()
            return
        except tk.TclError:
            state.modal_window = None

    master = state.root_tk
    win = tk.Toplevel(master) if master is not None else tk.Tk()
    state.modal_window = win
    win.title("实时模态/响应频率分析")
    win.geometry("840x620")
    win.minsize(620, 460)

    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    win.geometry(f"+{max(0, (sw - 840) // 2 + 110)}+{max(0, (sh - 620) // 2 - 40)}")

    p = state.modal_params
    vars_ = {
        "channel": tk.StringVar(value=CH_LABELS[p["channel"]]),
        "fft_size": tk.StringVar(value=str(p["fft_size"])),
        "target_freq": tk.StringVar(value=""),
        "x_min": tk.StringVar(value=f"{float(p.get('min_show_freq', 0.0)):g}"),
        "x_max": tk.StringVar(value=f"{float(p.get('max_freq', 100.0)):g}"),
    }

    # ---------------- 设置栏 ----------------
    cfg = tk.Frame(win)
    cfg.pack(fill=tk.X, padx=8, pady=(8, 2))

    tk.Label(cfg, text="分析通道", font=("SimHei", 9)).grid(
        row=0, column=0, sticky="e", padx=(4, 3), pady=3)
    cb_ch = ttk.Combobox(cfg, textvariable=vars_["channel"], values=CH_LABELS,
                         width=14, state="readonly")
    cb_ch.grid(row=0, column=1, sticky="w", pady=3)

    tk.Label(cfg, text="FFT点数", font=("SimHei", 9)).grid(
        row=0, column=2, sticky="e", padx=(18, 3), pady=3)
    cb_n = ttk.Combobox(cfg, textvariable=vars_["fft_size"], values=FFT_SIZES,
                        width=8, state="readonly")
    cb_n.grid(row=0, column=3, sticky="w", pady=3)

    def apply_settings(_event=None):
        try:
            target_freq = float(vars_["target_freq"].get())
            x_min = 0.0
            x_max = max(10.0, 2.0 * target_freq)
            vars_["x_min"].set(f"{x_min:g}")
            vars_["x_max"].set(f"{x_max:g}")
            vars_["target_freq"].set("")
        except ValueError:
            try:
                x_min = float(vars_["x_min"].get())
                x_max = float(vars_["x_max"].get())
            except ValueError:
                x_min = float(state.modal_params.get("min_show_freq", 0.0) or 0.0)
                x_max = float(state.modal_params.get("max_freq", 100.0) or 100.0)

        nyquist = float(state.ANALYSIS_RATE_HZ) / 2.0
        x_min = max(0.0, min(x_min, nyquist - 1e-6))
        x_max = max(x_min + 1.0, min(x_max, nyquist))
        vars_["x_min"].set(f"{x_min:g}")
        vars_["x_max"].set(f"{x_max:g}")

        state.modal_params.update({
            "channel": CH_LABELS.index(vars_["channel"].get()),
            "fft_size": int(vars_["fft_size"].get()),
            "min_show_freq": x_min,
            "max_freq": x_max,
            "revision": state.modal_params.get("revision", 0) + 1,
        })

    cb_ch.bind("<<ComboboxSelected>>", apply_settings)
    cb_n.bind("<<ComboboxSelected>>", apply_settings)

    tk.Label(cfg, text="目标Hz", font=("SimHei", 9)).grid(
        row=1, column=0, sticky="e", padx=(4, 3), pady=3)
    ent_target = ttk.Entry(cfg, textvariable=vars_["target_freq"], width=8)
    ent_target.grid(row=1, column=1, sticky="w", pady=3)

    tk.Label(cfg, text="X起点", font=("SimHei", 9)).grid(
        row=1, column=2, sticky="e", padx=(18, 3), pady=3)
    ent_xmin = ttk.Entry(cfg, textvariable=vars_["x_min"], width=8)
    ent_xmin.grid(row=1, column=3, sticky="w", pady=3)

    tk.Label(cfg, text="X终点", font=("SimHei", 9)).grid(
        row=1, column=4, sticky="e", padx=(18, 3), pady=3)
    ent_xmax = ttk.Entry(cfg, textvariable=vars_["x_max"], width=8)
    ent_xmax.grid(row=1, column=5, sticky="w", pady=3)

    btn_apply = ttk.Button(cfg, text="应用范围", command=apply_settings)
    btn_apply.grid(row=1, column=6, sticky="w", padx=(8, 4), pady=3)
    ent_target.bind("<Return>", apply_settings)
    ent_xmin.bind("<Return>", apply_settings)
    ent_xmax.bind("<Return>", apply_settings)

    # ---------------- 频谱图 ----------------
    fig = Figure(figsize=(6.4, 3.9), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("实时响应频谱 (去趋势 + 50Hz陷波 + 多帧平均)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (με)")
    ax.grid(True, alpha=0.3)

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    line, = ax.plot([], [], "-", color="#1976D2", lw=1.3)
    hint_text = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center",
                        fontsize=12, color="#888")

    # ---------------- 读数 ----------------
    info = tk.Frame(win)
    info.pack(fill=tk.X, padx=8, pady=(2, 8))

    sv_cur = tk.StringVar(value="当前响应峰(实时): --")
    sv_order = tk.StringVar(value="响应主频: --")
    sv_note = tk.StringVar(value="")
    sv_state = tk.StringVar(value="")

    tk.Label(info, textvariable=sv_cur, font=("Consolas", 11), fg="#C62828").pack(
        anchor="w", padx=2)
    order_label = tk.Label(info, textvariable=sv_order, font=("Consolas", 12, "bold"),
                           fg=COLOR_IDLE)
    order_label.pack(anchor="w", padx=2, pady=(2, 0))
    tk.Label(info, textvariable=sv_note, font=("SimHei", 9), fg="#777").pack(
        anchor="w", padx=2, pady=(3, 0))
    tk.Label(info, textvariable=sv_state, font=("SimHei", 9), fg=COLOR_WARN).pack(
        anchor="w", padx=2)

    # ---------------- 刷新循环 ----------------
    tick_ms = max(50, int(1000.0 / float(state.modal_params["update_hz"])))
    alive = [True]
    last_res = [None]
    y_top = [1.0]
    peak_annots = []

    def show_hint(text):
        hint_text.set_text(text)
        for a in peak_annots:
            try:
                a.remove()
            except Exception:
                pass
        peak_annots.clear()
        canvas.draw_idle()

    def refresh():
        res = state.modal_result
        if res is None:
            sv_state.set("状态: 正在采集数据…")
            show_hint("正在采集数据……")
            return
        if res is last_res[0]:
            return
        last_res[0] = res

        ar = res.get("analysis_rate_hz", 0.0) or 0.0   # 分析采样率 (频率轴依据)
        sr = res.get("sample_rate_hz", 0.0) or 0.0     # 仪器原始采样率 (固定)
        n_req = res.get("fft_size", 0)
        n_fed = res.get("n_fed", 0)
        df = res.get("df_hz", 0.0) or 0.0
        if ar > 0:
            if df > 0 and n_fed > 0:
                sv_note.set(f"分析采样率 {ar:.0f} Hz ({sr:.0f} Hz {state.DECIMATE}:1 抽取)"
                            f" | Δf = {df:.3f} Hz | 窗长 {n_fed / ar:.2f} s")
            else:
                sv_note.set(f"分析采样率 {ar:.0f} Hz ({sr:.0f} Hz {state.DECIMATE}:1 抽取)")

        status = res.get("status")
        if status == "collecting":
            sv_cur.set("当前响应峰(实时): --")
            sv_order.set("响应主频: --")
            order_label.config(fg=COLOR_IDLE)
            sv_state.set(f"状态: 正在采集数据…… ({res.get('n_have', 0)}/{res.get('n_need', 0)} 点)")
            show_hint("正在采集数据……")
            return
        if status == "waiting":
            sv_order.set("响应主频: --")
            order_label.config(fg=COLOR_IDLE)
            sv_state.set("状态: 等待数据…")
            show_hint("等待数据……")
            return

        freq_axis_hz, amps_avg = res.get("freq_axis_hz"), res.get("amps_avg")
        if freq_axis_hz is None or amps_avg is None:
            return

        x_min = float(state.modal_params.get("min_show_freq", 0.0) or 0.0)
        max_show = res.get("max_show", 0.0) or float(freq_axis_hz[-1])
        mask = (freq_axis_hz >= x_min) & (freq_axis_hz <= max_show)
        f_show = freq_axis_hz[mask]
        a_show = amps_avg[mask]
        line.set_data(f_show, a_show)

        cur_x = ax.get_xlim()
        if abs(cur_x[0] - x_min) > 1e-9 or abs(cur_x[1] - max_show) > 1e-9:
            ax.set_xlim(x_min, max_show)

        target = float(a_show.max()) * 1.25 if a_show.size else 1.0
        y_top[0] = target if target > y_top[0] else 0.9 * y_top[0] + 0.1 * target
        ax.set_ylim(0, max(y_top[0], 1e-9))

        # ---- 当前响应峰 (实时, 允许逐帧变化) ----
        response_cur = res.get("response_cur_hz")
        sv_cur.set(f"当前响应峰(实时): {response_cur:.2f} Hz" if response_cur
                   else "当前响应峰(实时): --")

        # ---- 响应主频 (多帧平均谱 + 主峰跟踪确认) ----
        hint_text.set_text("")
        for a in peak_annots:
            try:
                a.remove()
            except Exception:
                pass
        peak_annots.clear()

        o = res.get("response_peak")
        if o is None:
            sv_order.set("响应主频: -- (未检出)")
            order_label.config(fg=COLOR_IDLE)
            sv_state.set("状态: 实时分析 | 未检测到有效峰值")
        else:
            f_modal = o["freq_hz"]
            if a_show.size and x_min <= f_modal <= max_show:
                y_modal = float(np.interp(f_modal, f_show, a_show))
            else:
                y_modal = 0.0
            col = COLOR_OK if o["stable"] else COLOR_WARN
            if o["stable"]:
                sv_order.set(f"响应主频: {f_modal:.2f} Hz | 连续{o['n']}帧确认, "
                             f"波动 {o['spread_hz']:.2f} Hz | 稳定")
                sv_state.set("")
            else:
                sv_order.set(f"响应主频(识别中): {f_modal:.2f} Hz | 已跟踪 {o['n']} 帧")
                sv_state.set("")
            order_label.config(fg=col)
            if x_min <= f_modal <= max_show:
                peak_annots.append(ax.annotate(
                    f"主频 {f_modal:.2f} Hz" + ("" if o["stable"] else " (识别中)"),
                    xy=(f_modal, y_modal), xytext=(10, 26), textcoords="offset points",
                    fontsize=10, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9)))

        canvas.draw_idle()

    def tick():
        if not alive[0]:
            return
        try:
            refresh()
        except tk.TclError:
            return
        except Exception as e:
            print("模态窗口刷新错误:", e)
        try:
            win.after(tick_ms, tick)
        except tk.TclError:
            return

    def on_close():
        alive[0] = False
        state.modal_active = False
        state.modal_result = None
        modal_analysis.clear_buffer()
        try:
            canvas.get_tk_widget().destroy()
            win.destroy()
        except tk.TclError:
            pass
        state.modal_window = None

    win.protocol("WM_DELETE_WINDOW", on_close)
    modal_analysis.clear_buffer()   # 丢弃上次遗留的数据, 避免新旧序列混在一起
    state.modal_active = True
    tick()
