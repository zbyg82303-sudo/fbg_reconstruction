import struct
import socket as _socket
import numpy as np

SIGNATURE = 0x55AAAA55
CMD_SERIAL_DATA = 124

def recv_all(sock, size):
    data = b''
    while len(data) < size:
        try:
            packet = sock.recv(size - len(data))
        except _socket.timeout:
            return None
        if not packet:
            return None
        data += packet
    return data

def read_packet(sock):
    header = recv_all(sock, 12)
    if not header:
        return None
    signature, command, length = struct.unpack("<III", header)
    if signature != SIGNATURE:
        return None
    data = recv_all(sock, length)
    if data is None:
        return None
    return command, data

def parse_serial_data(data):
    offset = 0
    position = struct.unpack("<Q", data[offset:offset+8])[0]
    offset += 8
    data_count = struct.unpack("<I", data[offset:offset+4])[0]
    offset += 4
    channel_count = struct.unpack("<I", data[offset:offset+4])[0]
    offset += 4
    total = data_count * channel_count
    values = struct.unpack("<" + "f" * total, data[offset:offset+4*total])
    values = np.array(values).reshape(data_count, channel_count)
    return values

def read_dhdas_data(sock):
    while True:
        packet = read_packet(sock)
        if packet is None:
            return None
        command, data = packet
        if command != CMD_SERIAL_DATA:
            continue
        values = parse_serial_data(data)
        channel_count = values.shape[1]
        return -values[-1], channel_count