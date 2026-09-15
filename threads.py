import queue
import time

import numpy as np

import modal_analysis
import state
from dhdas import read_dhdas_data
from reconstruction import curvature_least_squares, curvature_reconstruction, frenet_reconstruction_arc
from beam_model import estimate_force, force_projection, theoretical_curve, beam_length


def _replace_queue_item(q, item):
    """队列只保留最新帧, 避免计算/绘图追赶旧数据造成延迟."""
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break
    try:
        q.put_nowait(item)
    except Exception:
        pass


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
    wait_t0 = 0.0
    first_data = True
    voltage_warned = False
    while True:
        try:
            result = read_dhdas_data(sock)
            if result is not None:
                raw, hw_channels, block = result
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

                # 实时模态分析: 送入本包全部采样行 (SAMPLE_RATE_HZ=10000 下的
                # 连续原始时域序列), 窗口未打开时该调用立即返回
                if state.modal_active:
                    modal_analysis.push_packet(block[:, valid_si], time.time())

                # 校验电压通道索引
                if vi < len(raw):
                    voltage = float(raw[vi])
                else:
                    if not voltage_warned:
                        voltage_warned = True
                        print(f"警告: 电压通道索引 {vi} 超出硬件通道范围 (共{len(raw)}通道), 电压置0")
                    voltage = 0.0

                _replace_queue_item(state.queue_data, (strain, voltage, time.time()))
            else:
                if no_data_count == 0:
                    print("等待DHDAS数据...")
                    wait_t0 = time.time()
                elif time.time() - wait_t0 >= 5.0:
                    print(f"DHDAS无数据 {time.time() - wait_t0:.0f}s")
                    wait_t0 = time.time()
                no_data_count += 1
            # 轮询间隔 2ms: 仪器采样率 SAMPLE_RATE_HZ=10000 下每包约 10ms 一个,
            # 需及时收完, 否则包会堆积/丢失, 模态分析拿到的时域序列会出现间隙.
            # 重建侧仍由 queue_data 满即丢, 行为不变.
            if result is None:
                time.sleep(0.002)
        except Exception as e:
            print("数据线程错误:", e)
            time.sleep(0.5)


def compute_thread():
    s = np.linspace(0, beam_length, state.points)
    gauge_positions = np.array(state.GAUGE_POSITIONS)
    eval_points = np.linspace(0, beam_length, 50)
    fine_x = np.arange(0, 301, 2)  # 0, 2, 4, ..., 300 mm
    # s 与测点位置固定: 力投影向量 phi 与测点单位理论挠度只算一次.
    # 理论挠度对 F 严格线性 (y = F·y1), 故测点理论值可直接用 y_gauge_unit 缩放.
    phi = force_projection(s)
    _, y_gauge_unit = theoretical_curve(1.0, gauge_positions)
    last_print = 0.0

    while True:
        try:
            try:
                strain, voltage, data_ts = state.queue_data.get(timeout=0.05)
            except queue.Empty:
                continue
            while not state.queue_data.empty():
                strain, voltage, data_ts = state.queue_data.get()
            strain = np.asarray(strain, dtype=float)
            state.current_strain_raw = strain

            # 清零修正
            offsets = state.strain_offsets
            if offsets is not None:
                n_off = min(len(offsets), len(strain))
                strain = strain.copy()
                strain[:n_off] -= offsets[:n_off]

            state.current_strain_ue = strain
            strain_for_log = strain[:4].copy()

            # 电压清零 + 位移计算
            voltage_zeroed = voltage - state.voltage_offset
            state.voltage_history.append(voltage_zeroed)
            avg_voltage = sum(state.voltage_history) / len(state.voltage_history)
            state.current_displacement_mm = avg_voltage / 100.0

            # 标定修正: ε_corrected = k · ε_measured + c
            if state.calibration_active and state.calibration_factors is not None:
                factors = state.calibration_factors
                n_cal = min(4, len(strain), len(factors))
                strain = strain.copy()
                fac = np.asarray(factors[:n_cal], dtype=float)
                strain[:n_cal] = fac[:, 0] * strain[:n_cal] + fac[:, 1]

            strain_si = strain * 1e-6

            coef = curvature_least_squares(strain_si)
            kappa_interp = curvature_reconstruction(s, coef)

            F_est = estimate_force(s, kappa_interp, phi)
            F_prev = state.F_prev
            diff = abs(F_est - F_prev)
            alpha = 0.3 if diff > 1.5 else 0.7
            F_current = alpha * F_prev + (1 - alpha) * F_est
            state.F_prev = F_current

            F_real = state.F_input["value"]

            x_rec, y_rec, _ = frenet_reconstruction_arc(s, kappa_interp)

            x_true_full, y_true_full = theoretical_curve(F_real, x_rec)

            y_rec_key = np.interp(gauge_positions, x_rec, y_rec)
            y_true_key = np.interp(gauge_positions, x_true_full, y_true_full)
            y_est_key = np.abs(F_est * y_gauge_unit)
            key_err = np.abs(y_rec_key - y_true_key)

            y_rec_eval = np.interp(eval_points, x_rec, y_rec)
            y_true_eval = np.interp(eval_points, x_true_full, y_true_full)
            err = y_rec_eval - y_true_eval
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err ** 2))
            y_scale = np.max(np.abs(y_true_eval))
            nrmse = rmse / (y_scale + 1e-6)
            accuracy = np.exp(-nrmse) * 100

            logger = state.logger
            if logger is not None and logger.collecting:
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
                    "fine_disp": np.abs(np.interp(fine_x, x_rec, y_rec)),
                })

            state.error_info_text = format_error_report(
                gauge_positions, mae, rmse, nrmse, accuracy,
                y_true_key, y_rec_key, y_est_key, key_err, F_est
            )

            now = time.time()
            if now - last_print >= 2.0:
                print("\n" + state.error_info_text)
                last_print = now

            _replace_queue_item(state.queue_result, (x_rec, y_rec, x_true_full, y_true_full, data_ts, voltage))

        except Exception as e:
            print(" 计算线程错误:", e)
