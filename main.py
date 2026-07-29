import threading
import time
import tkinter as tk

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib import gridspec

import state
from beam_model import generate_beam_mesh, generate_side_meshes, beam_length
from data_collection import DataCollection
from threads import data_thread, compute_thread
from ui_windows import open_cloud_map, update_cloud_map, open_error_analysis, update_error_window, open_calibration_window

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_app(sock, config):
    state.logger = DataCollection(target_count=100)
    state.load_calibration()  # 恢复上次保存的标定状态

    # 应用通道配置
    if config is not None:
        state.strain_channel_indices = config["strain_indices"]
        state.voltage_channel_index = config["voltage_idx"]
        print(f"通道映射 → 应变索引: {state.strain_channel_indices}, 电压索引: {state.voltage_channel_index}")
        print(f"  (应变4个 → 梁位置 60/120/180/240mm, 电压1个 → 位移=电压/100 mm)")
 
    # ---- UI 构建 ----
    plt.ion()
    fig = plt.figure("教学演示系统", figsize=(12, 7))

    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1.3], figure=fig)

    ax = fig.add_subplot(gs[0], projection='3d')
    ax_ui = fig.add_subplot(gs[1])
    ax_ui.axis('off')

    bbox = ax_ui.get_position()

    # ---- 初始梁网格 ----
    x_init = np.linspace(0, beam_length, state.points)
    y_init = np.zeros(state.points)
    X_m, Y_m, Zt_m, Zb_m, _, x_u, y_u, b_u, th_u = generate_beam_mesh(x_init, y_init, nx=60, nz=16)
    XL_m, YL_m, ZL_m, XR_m, YR_m, ZR_m = generate_side_meshes(x_u, y_u, b_u, th_u, nside=2)

    beam_color = (0.70, 0.73, 0.78)

    surfs = {}
    surfs['top'] = ax.plot_surface(X_m, Y_m, Zt_m, color=beam_color,
                                    alpha=0.85, shade=True, antialiased=True)
    surfs['bottom'] = ax.plot_surface(X_m, Y_m, Zb_m, color=beam_color,
                                       alpha=0.85, shade=True, antialiased=True)
    surfs['left'] = ax.plot_surface(XL_m, YL_m, ZL_m, color=beam_color,
                                     alpha=0.7, shade=True, antialiased=True)
    surfs['right'] = ax.plot_surface(XR_m, YR_m, ZR_m, color=beam_color,
                                      alpha=0.7, shade=True, antialiased=True)

    # ---- 高亮中心线 ----
    line_rec, = ax.plot([], [], [], '-', color='#00E676', linewidth=3.0,
                         label="Frenet 重建", zorder=10)
    line_theo, = ax.plot([], [], [], '--', color='#FF1744', linewidth=3.0,
                          label="理论曲线", zorder=10)

    # ---- 固定坐标轴 ----
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("变截面梁 3D 形变重建", fontsize=13, fontweight='bold')
    ax.set_xlim(*state.AXIS_XLIM)
    ax.set_ylim(*state.AXIS_YLIM)
    ax.set_zlim(*state.AXIS_ZLIM)
    try:
        ax.set_box_aspect([300, 56, 70])
    except Exception:
        pass
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc='upper right', fontsize=9)

    # ---- 延迟显示 (3D图区域右上方) ----
    ax_pos = ax.get_position()
    lat_w = 0.10
    lat_h = 0.14
    lat_x = ax_pos.x0 + ax_pos.width + 0.012
    lat_y = ax_pos.y0 + ax_pos.height - lat_h

    ax_latency = fig.add_axes((lat_x, lat_y, lat_w, lat_h))
    ax_latency.axis('off')
    ax_latency.text(0.5, 0.88, '系统延迟', transform=ax_latency.transAxes,
                    fontsize=8, fontweight='bold', ha='center', va='top', color='#888')
    latency_value_text = ax_latency.text(0.5, 0.52, '-- ms', transform=ax_latency.transAxes,
                                         fontsize=16, fontweight='bold', ha='center', va='center',
                                         family='Consolas', color='#999')
    latency_avg_text = ax_latency.text(0.5, 0.15, 'AVG -- ms', transform=ax_latency.transAxes,
                                       fontsize=8, ha='center', va='center',
                                       family='Consolas', color='#999')
    latency_bar_bg = plt.Rectangle((0.08, 0.35), 0.84, 0.06, transform=ax_latency.transAxes,
                                    facecolor='#e0e0e0', edgecolor='#bbb', linewidth=0.5, zorder=0)
    ax_latency.add_patch(latency_bar_bg)
    latency_bar_fill = plt.Rectangle((0.08, 0.35), 0.0, 0.06, transform=ax_latency.transAxes,
                                      facecolor='#4CAF50', edgecolor='none', zorder=1)
    ax_latency.add_patch(latency_bar_fill)

    # ---- 位移显示 (电压通道, 延迟显示下方) ----
    disp_h = 0.12
    disp_y = lat_y - disp_h - 0.015
    ax_disp = fig.add_axes((lat_x, disp_y, lat_w, disp_h))
    ax_disp.axis('off')
    ax_disp.text(0.5, 0.88, '电压位移', transform=ax_disp.transAxes,
                 fontsize=8, fontweight='bold', ha='center', va='top', color='#888')
    disp_value_text = ax_disp.text(0.5, 0.50, '-- mm', transform=ax_disp.transAxes,
                                    fontsize=14, fontweight='bold', ha='center', va='center',
                                    family='Consolas', color='#999')
    disp_avg_text = ax_disp.text(0.5, 0.13, 'AVG -- mm', transform=ax_disp.transAxes,
                                  fontsize=8, ha='center', va='center',
                                  family='Consolas', color='#999')

    # ---- 清零回调 ----
    def zero_strain(_event):
        if state.current_strain_raw is not None:
            state.strain_offsets = state.current_strain_raw.copy()
            print(f"清零完成: offsets = {state.strain_offsets}")
        else:
            print("清零失败: 暂无应变数据")
        # 电压清零: 取当前电压历史平均值作为偏移
        if len(state.voltage_history) > 0:
            state.voltage_offset = sum(state.voltage_history) / len(state.voltage_history)
            state.voltage_history.clear()
            print(f"电压清零完成: offset = {state.voltage_offset:.4f} V")
        else:
            state.voltage_offset = 0.0

    # ---- 控件布局 ----
    bx0 = bbox.x0
    by0 = bbox.y0
    bh = bbox.height
    bw = bbox.width

    # 输入区域 (左侧)
    input_left = bx0 + 0.01
    input_w = 0.10
    input_h = 0.40 * bh

    # Force 输入框
    def submit(text):
        try:
            val = float(text)
            state.F_input["value"] = val
        except ValueError:
            print("输入无效")

    axbox = fig.add_axes((input_left, by0 + 0.52 * bh, input_w, input_h))
    text_box = TextBox(axbox, 'Force (N)', initial="30")
    text_box.on_submit(submit)

    # 刷新率设置
    def set_refresh_rate(text):
        try:
            hz = float(text)
            hz = max(1, min(200, hz))
            state.REFRESH_HZ["value"] = hz
            interval_ms = int(1000.0 / hz)
            timer.interval = interval_ms
        except ValueError:
            pass

    ax_refresh = fig.add_axes((input_left, by0 + 0.04 * bh, input_w, input_h))
    refresh_box = TextBox(ax_refresh, '刷新(Hz)', initial="120")
    refresh_box.on_submit(set_refresh_rate)

    # 按钮区域 (右侧，均匀分布)
    n_buttons = 6
    btn_group_left = bx0 + 0.13
    btn_group_right = bx0 + bw - 0.005
    btn_area_w = btn_group_right - btn_group_left
    btn_spacing = btn_area_w / n_buttons
    btn_w = btn_spacing * 0.85
    btn_h = 0.70 * bh
    btn_y = by0 + 0.15 * bh

    def btn_x(i):
        return btn_group_left + i * btn_spacing + (btn_spacing - btn_w) / 2

    # 清零按钮
    ax_zero = fig.add_axes((btn_x(0), btn_y, btn_w, btn_h))
    zero_button = Button(ax_zero, '清零', color='#FF5722', hovercolor='#E64A19')
    zero_button.on_clicked(zero_strain)

    # 模式按钮
    def toggle_mode(_event):
        if state.mode["type"] == "manual":
            state.mode["type"] = "auto"
            button.label.set_text("Mode: Auto")
        else:
            state.mode["type"] = "manual"
            button.label.set_text("Mode: Manual")

    ax_button = fig.add_axes((btn_x(1), btn_y, btn_w, btn_h))
    button = Button(ax_button, 'Mode: Manual')
    button.on_clicked(toggle_mode)

    # 采集按钮
    def save_data(_event):
        state.logger.trigger_save()

    ax_save = fig.add_axes((btn_x(2), btn_y, btn_w, btn_h))
    save_button = Button(ax_save, '采集')
    save_button.on_clicked(save_data)

    # 位移云图按钮
    ax_cloud_btn = fig.add_axes((btn_x(3), btn_y, btn_w, btn_h))
    cloud_button = Button(ax_cloud_btn, '位移云图', color='#FF9800', hovercolor='#F57C00')
    cloud_button.on_clicked(open_cloud_map)

    # 误差分析按钮
    ax_error_btn = fig.add_axes((btn_x(4), btn_y, btn_w, btn_h))
    error_button = Button(ax_error_btn, '误差分析', color='#4CAF50', hovercolor='#388E3C')
    error_button.on_clicked(open_error_analysis)

    # 标定按钮
    ax_calib_btn = fig.add_axes((btn_x(5), btn_y, btn_w, btn_h))
    calib_button = Button(ax_calib_btn, '标定', color='#9C27B0', hovercolor='#7B1FA2')
    calib_button.on_clicked(open_calibration_window)

    # 获取 matplotlib TkAgg 的根窗口
    state.root_tk = fig.canvas.manager.window

    # ---- UI 刷新 ----
    zeros_line = np.zeros(state.points)
    frame_counter = [0]
    last_draw = [0.0]
    last_work = [0.0]
    pending = [None]

    def update_plot():
        # 始终排空队列，保持数据最新（开销极低）
        latest = None
        while not state.queue_result.empty():
            latest = state.queue_result.get()
        if latest is not None:
            pending[0] = latest

        # 节流实际工作到 ~50Hz，避免 120Hz 下 CPU/GPU 过载
        now = time.time()
        if now - last_work[0] < 0.020:
            return
        last_work[0] = now

        latest = pending[0]
        if latest is None:
            return
        pending[0] = None

        x_rec, y_rec, x_true, y_true, data_ts, _voltage = latest
        frame_counter[0] += 1

        # 曲面重建: 每 3 个工作周期 (~17Hz)
        if frame_counter[0] % 3 == 0:
            X_m, Y_m, Zt_m, Zb_m, _, x_u, y_u, b_u, th_u = \
                generate_beam_mesh(x_rec, y_rec, nx=60, nz=16)
            XL_m, YL_m, ZL_m, XR_m, YR_m, ZR_m = \
                generate_side_meshes(x_u, y_u, b_u, th_u, nside=2)

            for key in list(surfs.keys()):
                try:
                    surfs[key].remove()
                except Exception:
                    pass

            surfs['top'] = ax.plot_surface(X_m, Y_m, Zt_m, color=beam_color,
                                            alpha=0.85, shade=True, antialiased=True)
            surfs['bottom'] = ax.plot_surface(X_m, Y_m, Zb_m, color=beam_color,
                                               alpha=0.85, shade=True, antialiased=True)
            surfs['left'] = ax.plot_surface(XL_m, YL_m, ZL_m, color=beam_color,
                                             alpha=0.7, shade=True, antialiased=True)
            surfs['right'] = ax.plot_surface(XR_m, YR_m, ZR_m, color=beam_color,
                                              alpha=0.7, shade=True, antialiased=True)
            ax.set_xlim(*state.AXIS_XLIM)
            ax.set_ylim(*state.AXIS_YLIM)
            ax.set_zlim(*state.AXIS_ZLIM)

        # 中心线
        line_rec.set_data(x_rec, zeros_line)
        line_rec.set_3d_properties(y_rec + 2.0)
        line_theo.set_data(x_true, zeros_line)
        line_theo.set_3d_properties(y_true + 2.0)

        # Canvas 重绘 (draw_idle 已经是异步的，这里作为二层防护)
        if now - last_draw[0] >= 0.025:
            fig.canvas.draw_idle()
            last_draw[0] = now

        # 延迟显示: 每 3 个工作周期更新 (~17Hz，肉眼足够)
        if frame_counter[0] % 3 == 0:
            lat_ms = (time.time() - data_ts) * 1000
            state.latency_history.append(lat_ms)
            avg_ms = sum(state.latency_history) / len(state.latency_history)

            if lat_ms < 30:
                color = '#4CAF50'
            elif lat_ms < 80:
                color = '#FF9800'
            else:
                color = '#F44336'

            latency_value_text.set_text(f"{lat_ms:.0f} ms")
            latency_value_text.set_color(color)
            latency_avg_text.set_text(f"AVG {avg_ms:.0f} ms")
            latency_avg_text.set_color(color)

            bar_ratio = min(lat_ms / 200.0, 1.0)
            latency_bar_fill.set_width(bar_ratio * 0.84)
            latency_bar_fill.set_facecolor(color)

            # 位移显示更新
            disp_mm = state.current_displacement_mm
            disp_value_text.set_text(f"{disp_mm:.3f} mm")
            disp_value_text.set_color('#2196F3')
            disp_avg_text.set_text(f"AVG {disp_mm:.3f} mm")
            disp_avg_text.set_color('#2196F3')

        # 云图 / 误差窗口: 每个工作周期调用（内部有各自的节流）
        update_cloud_map(x_rec, y_rec, frame_counter[0])
        if frame_counter[0] % 2 == 0:  # ~25Hz
            update_error_window()

    # ---- 定时器 ----
    interval_ms = int(1000.0 / state.REFRESH_HZ["value"])
    timer = fig.canvas.new_timer(interval=interval_ms)
    timer.add_callback(update_plot)
    timer.start()

    # ---- 启动线程 ----
    t1 = threading.Thread(target=data_thread, args=(sock,), daemon=True)
    t2 = threading.Thread(target=compute_thread, daemon=True)
    t1.start()
    t2.start()

    print("系统启动完成")
    plt.show(block=True)

    if state.error_window is not None:
        try:
            state.error_window.destroy()
        except tk.TclError:
            pass
        state.error_window = None


if __name__ == "__main__":
    from ui_login import show_login
    result = show_login()
    if result is None or result[0] is None:
        print("用户取消登录")
    else:
        sock, config = result
        run_app(sock, config)
