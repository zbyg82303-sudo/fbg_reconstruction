import math
import numpy as np

sensor_positions = np.array([60, 120, 180, 240])

def curvature_least_squares(strain, sensor_y=1.75):
    n_top = min(len(strain), 4)
    strain_top = strain[:n_top]
    kappa = strain_top / sensor_y
    s = sensor_positions[:n_top]
    A = np.vstack([s**2, s, np.ones_like(s)]).T
    W = np.eye(n_top)
    if n_top > 1:
        W[1, 1] = 4
    W_sqrt = np.sqrt(W)
    coef = np.linalg.lstsq(W_sqrt @ A, W_sqrt @ kappa, rcond=None)[0]
    return coef


def curvature_reconstruction(s, coef):
    a, b, c = coef
    return a * s**2 + b * s + c


def frenet_reconstruction_arc(s, kappa):
    n = len(s)
    x = np.zeros(n)
    y = np.zeros(n)
    theta = 0.0

    for i in range(1, n):
        ds = s[i] - s[i - 1]
        k_avg = 0.5 * (kappa[i] + kappa[i - 1])
        dtheta = k_avg * ds

        if abs(k_avg) < 1e-8:
            dx = math.cos(theta) * ds
            dy = math.sin(theta) * ds
        else:
            sin_dt = math.sin(dtheta)
            cos_dt = math.cos(dtheta)
            sin_th = math.sin(theta)
            cos_th = math.cos(theta)
            inv_k = 1.0 / k_avg
            dx = (cos_th * sin_dt - sin_th * (1.0 - cos_dt)) * inv_k
            dy = (sin_th * sin_dt + cos_th * (1.0 - cos_dt)) * inv_k

        x[i] = x[i - 1] + dx
        y[i] = y[i - 1] + dy
        theta += dtheta

    return x, y, np.zeros(n)