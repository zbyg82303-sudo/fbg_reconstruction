import tkinter as tk
from tkinter import ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm

import state
from beam_model import generate_beam_mesh, generate_side_meshes, beam_length, E, h, b_func


# ===============================
# 位移云图
# ===============================
def open_cloud_map(_event=None):
    if state.cloud_fig is not None and plt.fignum_exists(state.cloud_fig.number):
        try:
            state.cloud_fig.canvas.manager.window.lift()
        except Exception:
            pass
        return

    state.cloud_fig = plt.figure("位移云图 — Total Deformation", figsize=(9, 7))
    state.cloud_ax = state.cloud_fig.add_subplot(111, projection='3d')
    state.cloud_ax.set_title("Total Deformation (位移云图)", fontsize=13, fontweight='bold')
    state.cloud_ax.set_xlabel("X (mm)")
    state.cloud_ax.set_ylabel("Y (mm)")
    state.cloud_ax.set_zlabel("Z (mm)")

    state.cloud_surfs = {}
    state.cloud_colorbar = None
    state.cloud_annotations = []

    state.cloud_ax.set_xlim(*state.AXIS_XLIM)
    state.cloud_ax.set_ylim(*state.AXIS_YLIM)
    state.cloud_ax.set_zlim(*state.AXIS_ZLIM)
    try:
        state.cloud_ax.set_box_aspect([300, 56, 70])
    except Exception:
        pass
    state.cloud_ax.view_init(elev=25, azim=-60)

    norm = PowerNorm(gamma=state.CLOUD_GAMMA, vmin=0, vmax=state.CLOUD_DMAX)
    mappable = cm.ScalarMappable(norm=norm, cmap='jet')
    mappable.set_array([])
    state.cloud_colorbar = state.cloud_fig.colorbar(mappable, ax=state.cloud_ax, shrink=0.6, pad=0.12,
                                                     ticks=state.COLOR_TICKS)
    state.cloud_colorbar.set_label("Displacement (mm)", fontsize=10)

    x_init = np.linspace(0, beam_length, 100)
    y_init = np.zeros(100)
    X, Y, Zt, Zb, _, x_u, y_u, b_u, th_u = generate_beam_mesh(x_init, y_init, nx=50, nz=20)
    X_l, Y_l, Z_l, X_r, Y_r, Z_r = generate_side_meshes(x_u, y_u, b_u, th_u, nside=2)

    beam_color = (0.70, 0.73, 0.78)
    state.cloud_surfs['top'] = state.cloud_ax.plot_surface(X, Y, Zt, color=beam_color,
                                                            alpha=0.85, shade=True, antialiased=True)
    state.cloud_surfs['bottom'] = state.cloud_ax.plot_surface(X, Y, Zb, color=beam_color,
                                                               alpha=0.85, shade=True, antialiased=True)
    state.cloud_surfs['left'] = state.cloud_ax.plot_surface(X_l, Y_l, Z_l, color=beam_color,
                                                             alpha=0.7, shade=True, antialiased=True)
    state.cloud_surfs['right'] = state.cloud_ax.plot_surface(X_r, Y_r, Z_r, color=beam_color,
                                                              alpha=0.7, shade=True, antialiased=True)

    state.cloud_fig.canvas.mpl_connect('close_event', on_cloud_close)
    state.cloud_fig.show()


def on_cloud_close(_event):
    state.cloud_fig = None
    state.cloud_ax = None
    state.cloud_surfs = {}
    state.cloud_colorbar = None
    state.cloud_annotations = []


def update_cloud_map(x_rec, y_rec, frame_count):
    if state.cloud_fig is None or not plt.fignum_exists(state.cloud_fig.number):
        return

    if frame_count % 4 != 0:
        return

    X, Y, Z_top, Z_bottom, disp_map, x_uni, y_uni, b_uni, theta_uni = \
        generate_beam_mesh(x_rec, y_rec, nx=50, nz=20)

    X_left, Y_left, Z_left, X_right, Y_right, Z_right = \
        generate_side_meshes(x_uni, y_uni, b_uni, theta_uni, nside=2)

    norm = PowerNorm(gamma=state.CLOUD_GAMMA, vmin=0, vmax=state.CLOUD_DMAX)

    for key in list(state.cloud_surfs.keys()):
        try:
            state.cloud_surfs[key].remove()
        except Exception:
            pass

    fc = plt.cm.jet(norm(disp_map))
    state.cloud_surfs['top'] = state.cloud_ax.plot_surface(X, Y, Z_top, facecolors=fc,
                                                            alpha=0.9, shade=True, antialiased=True)
    state.cloud_surfs['bottom'] = state.cloud_ax.plot_surface(X, Y, Z_bottom, facecolors=fc,
                                                               alpha=0.9, shade=True, antialiased=True)

    disp_side = np.tile(np.abs(y_uni[:, np.newaxis]), (1, 2))
    fc_side = plt.cm.jet(norm(disp_side))
    state.cloud_surfs['left'] = state.cloud_ax.plot_surface(X_left, Y_left, Z_left, facecolors=fc_side,
                                                             alpha=0.9, shade=True, antialiased=True)
    state.cloud_surfs['right'] = state.cloud_ax.plot_surface(X_right, Y_right, Z_right, facecolors=fc_side,
                                                              alpha=0.9, shade=True, antialiased=True)

    state.cloud_ax.set_xlim(*state.AXIS_XLIM)
    state.cloud_ax.set_ylim(*state.AXIS_YLIM)
    state.cloud_ax.set_zlim(*state.AXIS_ZLIM)

    for ann in state.cloud_annotations:
        try:
            ann.remove()
        except Exception:
            pass
    state.cloud_annotations = []

    for pos in state.GAUGE_POSITIONS:
        idx = np.argmin(np.abs(x_uni - pos))
        disp_val = abs(y_uni[idx])
        z_surface = y_uni[idx]
        state.cloud_annotations.append(
            state.cloud_ax.text(x_uni[idx], b_uni[idx] / 2 + 3, z_surface + 2,
                                f'{disp_val:.2f}',
                                color='black', fontsize=8, fontweight='bold',
                                ha='center', va='bottom')
        )
        state.cloud_annotations.append(
            state.cloud_ax.scatter(x_uni[idx], b_uni[idx] / 2, z_surface,
                                   c='black', s=20, marker='o', zorder=10)
        )

    state.cloud_fig.canvas.draw_idle()


# ===============================
# 误差分析窗口
# ===============================
def open_error_analysis(_event=None):
    if state.error_window is not None:
        try:
            state.error_window.lift()
            update_error_window()
            return
        except tk.TclError:
            state.error_window = None

    master = state.root_tk if state.root_tk is not None else tk.Tk()
    if state.root_tk is not None:
        state.error_window = tk.Toplevel(master)
    else:
        state.error_window = tk.Tk()

    state.error_window.title("误差分析")
    state.error_window.geometry("580x480")
    state.error_window.resizable(True, True)

    sw = state.error_window.winfo_screenwidth()
    sh = state.error_window.winfo_screenheight()
    x = (sw - 580) // 2 + 200
    y = (sh - 480) // 2
    state.error_window.geometry(f"+{x}+{y}")

    frame = tk.Frame(state.error_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    state.error_text_widget = tk.Text(frame, font=("Consolas", 10), wrap=tk.NONE,
                                      yscrollcommand=scrollbar.set)
    state.error_text_widget.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=state.error_text_widget.yview)

    def on_close():
        try:
            state.error_window.destroy()
        except tk.TclError:
            pass
        state.error_window = None
        state.error_text_widget = None

    state.error_window.protocol("WM_DELETE_WINDOW", on_close)
    update_error_window()


def update_error_window():
    if state.error_window is None or state.error_text_widget is None:
        return
    try:
        new_text = state.error_info_text or "等待数据..."
        current = state.error_text_widget.get(1.0, tk.END).rstrip("\n")
        if current == new_text:
            return
        state.error_text_widget.delete(1.0, tk.END)
        state.error_text_widget.insert(1.0, new_text)
        state.error_window.update_idletasks()
    except tk.TclError:
        pass


# ===============================
# 标定窗口 (应变修正)
# ===============================
def open_calibration_window(_event=None):
    if state.calibration_window is not None:
        try:
            state.calibration_window.lift()
            return
        except tk.TclError:
            state.calibration_window = None

    master = state.root_tk if state.root_tk is not None else tk.Tk()
    if state.root_tk is not None:
        state.calibration_window = tk.Toplevel(master)
    else:
        state.calibration_window = tk.Tk()

    state.calibration_window.title("标定窗口 — 应变修正")
    state.calibration_window.geometry("620x520")
    state.calibration_window.resizable(True, True)

    sw = state.calibration_window.winfo_screenwidth()
    sh = state.calibration_window.winfo_screenheight()
    x = (sw - 620) // 2 + 200
    y = (sh - 520) // 2
    state.calibration_window.geometry(f"+{x}+{y}")

    # ---- 输入区域 ----
    input_frame = tk.Frame(state.calibration_window)
    input_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

    tk.Label(input_frame, text="真实载荷 (N):", font=("SimHei", 10)).pack(side=tk.LEFT, padx=(0, 5))
    load_var = tk.StringVar(value="0")
    load_entry = tk.Entry(input_frame, textvariable=load_var, width=12, font=("Consolas", 11))
    load_entry.pack(side=tk.LEFT, padx=(0, 10))

    # 标定状态指示
    status_var = tk.StringVar(value="标定状态: 未激活")
    status_label = tk.Label(input_frame, textvariable=status_var, font=("SimHei", 10, "bold"),
                            fg="#999")
    status_label.pack(side=tk.LEFT, padx=(10, 0))

    # ---- 表格区域 ----
    tree_frame = tk.Frame(state.calibration_window)
    tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

    columns = ("#", "载荷(N)", "应变1(με)", "应变2(με)", "应变3(με)", "应变4(με)")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings",
                        selectmode="browse", height=10)
    col_widths = [40, 90, 110, 110, 110, 110]
    for col, width in zip(columns, col_widths):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="center")

    scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ---- 拟合结果显示 ----
    result_frame = tk.Frame(state.calibration_window)
    result_frame.pack(fill=tk.X, padx=8, pady=(2, 6))

    result_text = tk.Text(result_frame, font=("Consolas", 9), height=8, wrap=tk.NONE,
                          state=tk.DISABLED, bg="#f5f5f5")
    result_text.pack(fill=tk.X)

    # ---- 按钮区域 ----
    btn_frame = tk.Frame(state.calibration_window)
    btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

    # ---- 回调函数 ----
    def do_record():
        try:
            load_val = float(load_var.get())
        except ValueError:
            _show_result("请输入有效的数值载荷")
            return
        if state.current_strain_ue is None:
            _show_result("暂无应变数据，请等待数据采集")
            return
        s = state.current_strain_ue[:4]
        state.calibration_data.append((load_val, s.tolist()))
        state.save_calibration()
        _refresh_tree()

    def do_delete():
        sel = tree.selection()
        if not sel:
            return
        idx = tree.index(sel[0])
        if 0 <= idx < len(state.calibration_data):
            del state.calibration_data[idx]
            state.save_calibration()
            _refresh_tree()
            _show_result("")

    def do_fit():
        if len(state.calibration_data) < 2:
            _show_result("至少需要两组数据才能拟合")
            return
        loads = np.array([d[0] for d in state.calibration_data])
        strains = np.array([d[1] for d in state.calibration_data])  # (n, 4)

        positions = np.array([60, 120, 180, 240], dtype=float)

        # 理论灵敏度 K_theory (με/N)，用于计算各通道理论应变
        sensor_y = 1.75
        b_vals = b_func(positions)
        I_vals = b_vals * h**3 / 12
        K_theory = (beam_length - positions) * sensor_y * 1e6 / (E * I_vals)

        lines = ["拟合结果 (ε_theory = k · ε_measured + c):", "-" * 58]
        coeffs_all = []
        for i in range(4):
            eps_meas = strains[:, i]
            eps_theory = -K_theory[i] * loads   # 理论应变

            k, c = np.polyfit(eps_meas, eps_theory, 1)
            eps_pred = k * eps_meas + c
            ss_res = np.sum((eps_theory - eps_pred) ** 2)
            ss_tot = np.sum((eps_theory - eps_theory.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            lines.append(f"通道{i+1} ({positions[i]}mm): "
                         f"ε_theory = {k:.4f} · ε_measured + {c:.4f}   (R²={r2:.4f})")
            coeffs_all.append((k, c))
        lines.append("-" * 58)
        state.calibration_coeffs = coeffs_all

        # 修正系数即为拟合系数: ε_corrected = k · ε_measured + c
        factors = list(coeffs_all)
        lines.append("")
        lines.append("修正公式 (已激活 ✓): ε_corrected = k · ε_measured + c")
        lines.append("-" * 58)
        for i in range(4):
            k_i, c_i = factors[i]
            lines.append(f"通道{i+1} ({positions[i]}mm): "
                         f"k = {k_i:.4f} | c = {c_i:.2f} με")
        lines.append("-" * 58)
        lines.append("提示: 修正已自动激活，应变数据将在重建前进行校正。")
        lines.append("      点击「清除标定」可停用修正。")

        state.calibration_factors = factors
        state.calibration_active = True
        state.save_calibration()
        _update_status()
        _show_result("\n".join(lines))

    def do_clear():
        state.calibration_active = False
        state.calibration_factors = None
        state.save_calibration()
        _update_status()
        _show_result("标定已清除，重建恢复默认模式。")

    def _update_status():
        if state.calibration_active:
            status_var.set("标定状态: 已激活 ✓")
            status_label.config(fg="#4CAF50")
        else:
            status_var.set("标定状态: 未激活")
            status_label.config(fg="#999")

    def _show_result(text):
        result_text.configure(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(1.0, text)
        result_text.configure(state=tk.DISABLED)

    def _refresh_tree():
        for item in tree.get_children():
            tree.delete(item)
        for i, (load, ss) in enumerate(state.calibration_data):
            tree.insert("", tk.END, values=(i, f"{load:.2f}",
                        f"{ss[0]:.2f}", f"{ss[1]:.2f}", f"{ss[2]:.2f}", f"{ss[3]:.2f}"))

    def on_tree_delete(_event):
        do_delete()

    tree.bind("<Delete>", on_tree_delete)

    tk.Button(btn_frame, text="记录", command=do_record,
              font=("SimHei", 10, "bold"), bg="#2196F3", fg="white",
              width=8, height=1).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_frame, text="删除", command=do_delete,
              font=("SimHei", 10, "bold"), bg="#FF9800", fg="white",
              width=8, height=1).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_frame, text="拟合", command=do_fit,
              font=("SimHei", 10, "bold"), bg="#4CAF50", fg="white",
              width=8, height=1).pack(side=tk.LEFT, padx=(0, 6))
    tk.Button(btn_frame, text="清除标定", command=do_clear,
              font=("SimHei", 10, "bold"), bg="#F44336", fg="white",
              width=10, height=1).pack(side=tk.LEFT)

    def on_close():
        for item in tree.get_children():
            tree.delete(item)
        try:
            state.calibration_window.destroy()
        except tk.TclError:
            pass
        state.calibration_window = None

    state.calibration_window.protocol("WM_DELETE_WINDOW", on_close)

    # 恢复已有数据
    _refresh_tree()
    _update_status()
    if state.calibration_coeffs is not None and state.calibration_factors is not None:
        positions = np.array([60, 120, 180, 240], dtype=float)
        lines = ["拟合结果 (ε_theory = k · ε_measured + c):", "-" * 58]
        for i, (k, c) in enumerate(state.calibration_coeffs):
            lines.append(f"通道{i+1} ({positions[i]}mm): "
                         f"ε_theory = {k:.4f} · ε_measured + {c:.4f}")
        lines.append("-" * 58)
        lines.append("")
        status = " (已激活 ✓):" if state.calibration_active else " (未激活):"
        lines.append("修正公式" + status + " ε_corrected = k · ε_measured + c")
        lines.append("-" * 58)
        for i, (k_i, c_i) in enumerate(state.calibration_factors):
            lines.append(f"通道{i+1} ({positions[i]}mm): "
                         f"k = {k_i:.4f} | c = {c_i:.2f} με")
        lines.append("-" * 58)
        _show_result("\n".join(lines))
