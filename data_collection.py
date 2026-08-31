import csv
import datetime

import numpy as np


class DataCollection:
    def __init__(self, target_count=100):
        self.filename = None
        self.file = None
        self.writer = None
        self.collecting = False
        self.count = 0
        self.target_count = target_count
        self.buffer = []
        self.fine_disp_buffer = []
        self.collection_round = 0

    def _ensure_file(self):
        if self.file is None:
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = f"dataset_{now}.csv"
            self.file = open(self.filename, "w", newline="", encoding="utf-8-sig")
            self.writer = csv.writer(self.file)
            header = [
                "序号", "真实力(N)", "反演力(N)",
                "应变1(με)", "应变2(με)", "应变3(με)", "应变4(με)",
                "位移60mm", "位移120mm", "位移180mm", "位移240mm", "位移300mm",
                "反演力位移60mm", "反演力位移120mm", "反演力位移180mm", "反演力位移240mm", "反演力位移300mm",
                "理论位移60mm", "理论位移120mm", "理论位移180mm", "理论位移240mm", "理论位移300mm",
                "电压位移(mm)", "原始电压(V)",
                "MAE", "RMSE", "NRMSE", "Accuracy"
            ]
            self.writer.writerow(header)
            print(f"数据文件创建: {self.filename}")

    def trigger_save(self):
        if self.collecting:
            print("已在采集中，请勿重复触发")
            return
        self._ensure_file()
        self.collecting = True
        self.count = 0
        self.buffer = []
        self.fine_disp_buffer = []
        self.collection_round += 1
        print(f"开始采集第{self.collection_round}组（目标{self.target_count}组）")

    def log(self, data):
        if not self.collecting or self.writer is None:
            return

        self.count += 1
        strains = data["strains"]
        disp = data["displacements"]
        disp_est = data.get("disp_est", [0, 0, 0, 0, 0])
        disp_theory = data.get("disp_theory", [0, 0, 0, 0, 0])

        disp_mm = data.get("displacement_mm", 0.0)
        voltage_raw = data.get("voltage_raw", 0.0)
        row = [
            self.count,
            data["F_real"],
            data["F_est"],
            strains[0], strains[1], strains[2], strains[3],
            disp[0], disp[1], disp[2], disp[3], disp[4],
            disp_est[0], disp_est[1], disp_est[2], disp_est[3], disp_est[4],
            disp_theory[0], disp_theory[1], disp_theory[2], disp_theory[3], disp_theory[4],
            disp_mm,
            voltage_raw,
            data["mae"],
            data["rmse"],
            data["nrmse"],
            data["accuracy"],
        ]
        self.writer.writerow(row)
        self.buffer.append(data)

        fine_disp = data.get("fine_disp")
        if fine_disp is not None:
            self.fine_disp_buffer.append(fine_disp)

        if self.count % 10 == 0:
            self.file.flush()

        if self.count >= self.target_count:
            self._finalize_collection()

    def _finalize_collection(self):
        self.collecting = False
        self.file.flush()

        if not self.buffer:
            return

        F_real = self.buffer[0]["F_real"]
        F_est_arr = np.array([d["F_est"] for d in self.buffer])
        disp_arr = np.array([d["displacements"] for d in self.buffer])
        disp_est_arr = np.array([d.get("disp_est", [0, 0, 0, 0, 0]) for d in self.buffer])
        disp_theory_arr = np.array([d.get("disp_theory", [0, 0, 0, 0, 0]) for d in self.buffer])
        mae_arr = np.array([d["mae"] for d in self.buffer])
        rmse_arr = np.array([d["rmse"] for d in self.buffer])
        nrmse_arr = np.array([d["nrmse"] for d in self.buffer])
        acc_arr = np.array([d["accuracy"] for d in self.buffer])

        self.writer.writerow([])
        self.writer.writerow([f"--- 采集组 #{self.collection_round} | 真实力: {F_real}N | 统计摘要 ---"])
        self.writer.writerow(["指标", "均值", "标准差"])

        gauge_labels = ["60mm", "120mm", "180mm", "240mm", "300mm"]

        self.writer.writerow(["反演力(N)",
                              f"{np.mean(F_est_arr):.4f}",
                              f"{np.std(F_est_arr):.4f}"])
        for i, label in enumerate(gauge_labels):
            self.writer.writerow([f"位移{label}",
                                  f"{np.mean(disp_arr[:, i]):.4f}",
                                  f"{np.std(disp_arr[:, i]):.4f}"])
        for i, label in enumerate(gauge_labels):
            self.writer.writerow([f"反演力位移{label}",
                                  f"{np.mean(disp_est_arr[:, i]):.4f}",
                                  f"{np.std(disp_est_arr[:, i]):.4f}"])
        for i, label in enumerate(gauge_labels):
            self.writer.writerow([f"理论位移{label}",
                                  f"{np.mean(disp_theory_arr[:, i]):.4f}",
                                  f"{np.std(disp_theory_arr[:, i]):.4f}"])
        self.writer.writerow(["MAE",
                              f"{np.mean(mae_arr):.4f}",
                              f"{np.std(mae_arr):.4f}"])
        self.writer.writerow(["RMSE",
                              f"{np.mean(rmse_arr):.4f}",
                              f"{np.std(rmse_arr):.4f}"])
        self.writer.writerow(["NRMSE",
                              f"{np.mean(nrmse_arr):.4f}",
                              f"{np.std(nrmse_arr):.4f}"])
        self.writer.writerow(["Accuracy",
                              f"{np.mean(acc_arr):.4f}",
                              f"{np.std(acc_arr):.4f}"])
        self.writer.writerow([])

        # ---- 细粒度位移 (0~300mm, 步长2mm, 100组平均) ----
        if self.fine_disp_buffer:
            fine_avg = np.mean(self.fine_disp_buffer, axis=0)
            fine_x = np.arange(0, 301, 2)
            header = [f"{x}mm位移" for x in fine_x]
            self.writer.writerow(header)
            self.writer.writerow([f"{v:.4f}" for v in fine_avg])
            self.writer.writerow([])

        self.file.flush()
        print(f"第{self.collection_round}组采集完成！统计摘要已保存")

    def close(self):
        if self.file:
            self.file.close()
            print("数据文件已关闭")
