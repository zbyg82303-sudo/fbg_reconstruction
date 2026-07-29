import socket
import tkinter as tk


def show_login():
    root = tk.Tk()
    root.title("教学演示系统")
    root.geometry("460x480")
    root.resizable(False, False)

    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - 460) // 2
    y = (sh - 480) // 2
    root.geometry(f"+{x}+{y}")

    tk.Label(root, text="教学演示系统", font=("SimHei", 16, "bold")).pack(pady=(20, 12))

    frm = tk.Frame(root)
    frm.pack(pady=5)

    row = 0

    # IP
    tk.Label(frm, text="IP 地址:", font=("SimHei", 11)).grid(row=row, column=0, sticky="e", padx=(0, 8))
    ip_var = tk.StringVar(value="192.168.43.181")
    tk.Entry(frm, textvariable=ip_var, width=24, font=("Consolas", 11)).grid(row=row, column=1)
    row += 1

    # Port
    tk.Label(frm, text="端口号:", font=("SimHei", 11)).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=(10, 0))
    port_var = tk.StringVar(value="50000")
    tk.Entry(frm, textvariable=port_var, width=24, font=("Consolas", 11)).grid(row=row, column=1, pady=(10, 0))
    row += 1

    # DHDAS output channel list
    tk.Label(frm, text="输出通道:", font=("SimHei", 11)).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=(14, 0))
    output_var = tk.StringVar(value="1,2,5,11,12")
    tk.Entry(frm, textvariable=output_var, width=24, font=("Consolas", 11)).grid(row=row, column=1, pady=(14, 0))
    tk.Label(frm, text="(DHDAS输出的全部通道, 逗号分隔, 按输出顺序)", fg="gray", font=("SimHei", 8)).grid(row=row + 1, column=1, sticky="w")
    row += 2

    # Strain channels
    tk.Label(frm, text="应变通道:", font=("SimHei", 11)).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=(10, 0))
    strain_var = tk.StringVar(value="1,2,11,12")
    tk.Entry(frm, textvariable=strain_var, width=24, font=("Consolas", 11)).grid(row=row, column=1, pady=(10, 0))
    tk.Label(frm, text="(4个, 从输出通道中选, 对应梁位置60/120/180/240mm)", fg="gray", font=("SimHei", 8)).grid(row=row + 1, column=1, sticky="w")
    row += 2

    # Voltage channel
    tk.Label(frm, text="电压通道:", font=("SimHei", 11)).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=(10, 0))
    voltage_var = tk.StringVar(value="5")
    tk.Entry(frm, textvariable=voltage_var, width=24, font=("Consolas", 11)).grid(row=row, column=1, pady=(10, 0))
    tk.Label(frm, text="(1个, 从输出通道中选, 不参与形状重建)", fg="gray", font=("SimHei", 8)).grid(row=row + 1, column=1, sticky="w")
    row += 2

    status_var = tk.StringVar(value="")
    tk.Label(root, textvariable=status_var, fg="gray", font=("SimHei", 9)).pack(pady=(8, 0))

    result = {"sock": None, "config": None}

    def parse_channels():
        """Return (strain_indices_in_raw, voltage_idx_in_raw) or None on error.
        Uses the output channel list to map channel numbers to raw data positions."""

        # 1. 解析输出通道列表
        try:
            output_list = [int(x.strip()) for x in output_var.get().strip().split(",")]
        except ValueError:
            status_var.set("输出通道格式错误，请输入数字")
            return None
        if len(output_list) < 5:
            status_var.set(f"输出通道至少需要5个 (4应变+1电压), 当前{len(output_list)}个")
            return None
        if len(set(output_list)) != len(output_list):
            status_var.set("输出通道不能重复")
            return None

        # 2. 解析应变通道
        try:
            strain_channels = [int(x.strip()) for x in strain_var.get().strip().split(",")]
        except ValueError:
            status_var.set("应变通道格式错误，请输入4个数字")
            return None
        if len(strain_channels) != 4:
            status_var.set(f"应变通道需要4个，当前输入了{len(strain_channels)}个")
            return None
        if len(set(strain_channels)) != 4:
            status_var.set("应变通道不能重复")
            return None

        # 3. 解析电压通道
        try:
            voltage_ch = int(voltage_var.get().strip())
        except ValueError:
            status_var.set("电压通道格式错误，请输入1个数字")
            return None

        # 4. 校验所有通道都在输出列表中
        for ch in strain_channels:
            if ch not in output_list:
                status_var.set(f"应变通道 {ch} 不在输出通道列表中")
                return None
        if voltage_ch not in output_list:
            status_var.set(f"电压通道 {voltage_ch} 不在输出通道列表中")
            return None
        if voltage_ch in strain_channels:
            status_var.set(f"电压通道 {voltage_ch} 与应变通道重复")
            return None

        # 5. 映射: 通道号 → 在原始数据中的位置
        strain_indices = [output_list.index(ch) for ch in strain_channels]
        voltage_idx = output_list.index(voltage_ch)

        return strain_indices, voltage_idx

    def do_connect():
        ch = parse_channels()
        if ch is None:
            return
        strain_indices, voltage_idx = ch

        ip = ip_var.get().strip()
        port_str = port_var.get().strip()
        if not ip or not port_str:
            status_var.set("请输入 IP 地址和端口号")
            return
        try:
            port = int(port_str)
        except ValueError:
            status_var.set("端口号必须是整数")
            return

        status_var.set(f"正在连接 {ip}:{port} ...")
        connect_btn.config(state="disabled")
        root.update()

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((ip, port))
            s.settimeout(2)
            result["sock"] = s
            result["config"] = {
                "strain_indices": strain_indices,
                "voltage_idx": voltage_idx,
            }
            root.destroy()
        except socket.timeout:
            status_var.set("连接超时，请检查 IP 和端口")
            connect_btn.config(state="normal")
        except ConnectionRefusedError:
            status_var.set("连接被拒绝，请检查 DHDAS 是否运行")
            connect_btn.config(state="normal")
        except Exception as e:
            status_var.set(f"连接失败: {e}")
            connect_btn.config(state="normal")

    connect_btn = tk.Button(root, text="连 接", command=do_connect,
                            font=("SimHei", 12, "bold"), bg="#2196F3", fg="white",
                            width=14, height=1, cursor="hand2")
    connect_btn.pack(pady=(12, 0))

    def on_enter(_event):
        do_connect()
    root.bind("<Return>", on_enter)

    root.mainloop()
    return result["sock"], result["config"]
