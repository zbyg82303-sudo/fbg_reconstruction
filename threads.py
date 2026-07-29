import time

import numpy as np
from scipy.interpolate import CubicSpline

import state
from dhdas import read_dhdas_data
from reconstruction import curvature_least_squares, curvature_reconstruction, frenet_reconstruction_arc
from beam_model import estimate_force, theoretical_curve, beam_length


def format_error_report(key_points, mae, rmse, nrmse, accuracy,
                        y_true_key, y_rec_key, y_est_key, key_err, F_est):
    lines = []
    lines.append("=" * 50)
    lines.append(f"模式: {state.mode['type']}")
    lines.append("")
    lines.append("误差评价指标")
    lines.append("-" * 40)
    lines.append(f"{'指标':12s} | {'数值':>12s}")
    lines.append("-" * 40)
    lines.append(f"{'MAE':12s} | {mae:8.3f} mm")
    lines.append(f"{'RMSE':12s} | {rmse:8.3f} mm")
    lines.append(f"{'NRMSE':12s} | {nrmse:8.4f}")
    lines.append(f"{'Accuracy':12s} | {accuracy:7.2f} %")
    lines.append("-" * 40)
    lines.append("")
    lines.append("关键点误差分析 (单位: mm)")
    lines.append("-" * 80)
    lines.append(f"{'位置(mm)':>8s} | {'真实挠度':>12s} | {'Frenet挠度':>12s} | {'反演力挠度':>12s} | {'差值':>10s}")
    lines.append("-" * 80)
    for i in range(len(key_points)):
        lines.append(f"{key_points[i]:8.0f} | "
                     f"{y_true_key[i]:12.3f} | "
                     f"{y_rec_key[i]:12.3f} | "
                     f"{y_est_key[i]:12.3f} | "
                     f"{key_err[i]:10.3f}")
    lines.append("-" * 80)
    lines.append(f"力估计: {F_est:.2f} N")
    lines.append("=" * 50)
    return "\n".join(lines)


def data_thread(sock):
    no_data_count = 0
    first_data = True
    voltage_warned = False
    while True:
        try:
            result = read_dhdas_data(sock)
            if result is not None:
                raw, hw_channels = result
                no_data_count = 0

                if first_data:
                    first_data = False
                    print(f"DHDAS 数据已连接, 硬件通道数: {hw_channels}, 原始数据长度: {len(raw)}")
                    print(f"  配置应变通道索引: {state.strain_channel_indices}")
                    print(f"  配置电压通道索引: {state.voltage_channel_index}")

                si = state.strain_channel_indices
                vi = state.voltage_channel_index

                # 校验应变通道索引
                valid_si = [i for i in si if i < len(raw)]
                if len(valid_si) < len(si):
                    bad = [i for i in si if i >= len(raw)]
                    print(f"警告: 应变通道索引 {bad} 超出硬件通道范围 (共{len(raw)}通道), 已忽略")
                if len(valid_si) == 0:
                    print("错误: 所有应变通道索引均无效, 跳过本帧")
                    time.sleep(0.01)
                    continue

                strain = raw[valid_si].copy()

                # 校验电压通道索引
                if vi < len(raw):
                    voltage = float(raw[vi])
                else:
                    if not voltage_warned:
                        voltage_warned = True
                        print(f"警告: 电压通道索引 {vi} 超出硬件通道范围 (共{len(raw)}通道), 电压置0")
                    voltage = 0.0

                if not state.queue_data.full():
                    state.queue_data.put((strain, voltage, time.time()))
            else:
                no_data_count += 1
                if no_data_count == 1:
                    print("等待DHDAS数据...")
                elif no_data_count % 100 == 0:
                    print(f"DHDAS无数据 {no_data_count * 0.01:.0f}s")
            time.sleep(0.01)
        except Exception as e:
            print("数据线程错误:", e)
            time.sleep(0.5)


def compute_thread():
    s = np.linspace(0, beam_length, state.points)
    gauge_positions = np.array(state.GAUGE_POSITIONS)
    eval_points = np.linspace(0, beam_length, 50)
    last_print = 0.0

    while True:
        try:
            if state.queue_data.empty():
                time.sleep(0.005)
                continue

            strain, voltage, data_ts = state.queue_data.get()
            state.current_strain_raw = strain.copy()

            # 清零修正
            offsets = state.strain_offsets
            if offsets is not None:
                n_off = min(len(offsets), len(strain))
                for i in range(n_off):
                    strain[i] -= offsets[i]

            state.current_strain_ue = strain.copy()
            strain_for_log = strain[:4].copy()

            # 电压清零 + 位移计算
            voltage_zeroed = voltage - state.voltage_offset
            state.voltage_history.append(voltage_zeroed)
            avg_voltage = sum(state.voltage_history) / len(state.voltage_history)
            state.current_displacement_mm = avg_voltage / 100.0

            # 标定修正: ε_corrected = k · ε_measured + c
            if state.calibration_active and state.calibration_factors is not None:
                factors = state.calibration_factors
                for i in range(min(4, len(strain))):
                    k_i, c_i = factors[i]
                    strain[i] = k_i * strain[i] + c_i

            strain_si = strain * 1e-6

            coef = curvature_least_squares(strain_si)
            kappa_interp = curvature_reconstruction(s, coef)

            F_est = estimate_force(s, kappa_interp)
            F_prev = state.F_prev
            diff = abs(F_est - F_prev)
            alpha = 0.3 if diff > 1.5 else 0.7
            F_current = alpha * F_prev + (1 - alpha) * F_est
            state.F_prev = F_current

            F_used = state.F_input["value"] if state.mode["type"] == "manual" else F_current
            F_real = state.F_input["value"]

            x_rec, y_rec, _ = frenet_reconstruction_arc(s, kappa_interp)
            cs_rec = CubicSpline(x_rec, y_rec)

            x_true_full, y_true_full = theoretical_curve(F_real, x_rec)
            cs_true = CubicSpline(x_true_full, y_true_full)

            y_rec_key = cs_rec(gauge_positions)
            y_true_key = cs_true(gauge_positions)
            _, y_est_key = theoretical_curve(F_est, gauge_positions)
            y_est_key = np.abs(y_est_key)
            key_err = np.abs(y_rec_key - y_true_key)

            y_true_eval = cs_true(eval_points)
            err = cs_rec(eval_points) - y_true_eval
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err ** 2))
            y_scale = np.max(np.abs(y_true_eval))
            nrmse = rmse / (y_scale + 1e-6)
            accuracy = np.exp(-nrmse) * 100

            logger = state.logger
            if logger is not None:
                logger.log({
                    "F_real": F_real,
                    "F_est": F_est,
                    "strains": strain_for_log,
                    "displacements": np.abs(y_rec_key),
                    "disp_theory": np.abs(y_true_key),
                    "disp_est": y_est_key,
                    "mae": mae,
                    "rmse": rmse,
                    "nrmse": nrmse,
                    "accuracy": accuracy,
                    "displacement_mm": state.current_displacement_mm,
                    "voltage_raw": voltage,
                })

            state.error_info_text = format_error_report(
                gauge_positions, mae, rmse, nrmse, accuracy,
                y_true_key, y_rec_key, y_est_key, key_err, F_est
            )

            now = time.time()
            if now - last_print >= 2.0:
                print("\n" + state.error_info_text)
                last_print = now

            while not state.queue_result.empty():
                state.queue_result.get()
            state.queue_result.put((x_rec, y_rec, x_true_full, y_true_full, data_ts, voltage))

            time.sleep(0.01)

        except Exception as e:
            print(" 计算线程错误:", e)
