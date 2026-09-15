import socket as _socket
import struct

import numpy as np

SIGNATURE = 0x55AAAA55
CMD_SERIAL_DATA = 124
_HEADER_STRUCT = struct.Struct("<III")
_SERIAL_PREFIX_STRUCT = struct.Struct("<QII")

def recv_all(sock, size):
    chunks = []
    received = 0
    while received < size:
        try:
            packet = sock.recv(size - received)
        except _socket.timeout:
            return None
        if not packet:
            return None
        chunks.append(packet)
        received += len(packet)
    return chunks[0] if len(chunks) == 1 else b''.join(chunks)

def read_packet(sock):
    header = recv_all(sock, 12)
    if not header:
        return None
    signature, command, length = _HEADER_STRUCT.unpack(header)
    if signature != SIGNATURE:
        return None
    data = recv_all(sock, length)
    if data is None:
        return None
    return command, data

def parse_serial_data(data):
    _position, data_count, channel_count = _SERIAL_PREFIX_STRUCT.unpack_from(data, 0)
    offset = _SERIAL_PREFIX_STRUCT.size
    total = data_count * channel_count
    values = np.frombuffer(data, dtype="<f4", count=total, offset=offset)
    values = values.reshape(data_count, channel_count)
    return values

def read_dhdas_data(sock):
    """读取一个采样包.

    返回 (最后一行, 通道数, 整包数据) 或 None.
    最后一行 = 原代码的返回值 (符号已取反), 整包数据形状 (data_count, channel_count),
    供需要完整时域序列的模块使用 (如实时模态分析).
    """
    while True:
        packet = read_packet(sock)
        if packet is None:
            return None
        command, data = packet
        if command != CMD_SERIAL_DATA:
            continue
        values = parse_serial_data(data)
        channel_count = values.shape[1]
        block = -values
        return block[-1], channel_count, block
