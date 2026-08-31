import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

beam_length = 300
E = 2e5
h = 3.5

# ===============================
# 截面函数
# ===============================
def b_func(x):
    x = np.array(x, dtype=float)
    b = np.zeros_like(x, dtype=float)

    mask1 = x <= 240
    b[mask1] = 46 - (46 - 9)/240 * x[mask1]

    mask2 = x > 240
    b[mask2] = 9

    return b

def I_func(x):
    return b_func(x) * h**3 / 12


# ===============================
# 理论曲线
# ===============================
def theoretical_curve(F, x_ref=None):

    # 用同一个网格（避免插值误差）
    if x_ref is None:
        x = np.linspace(0, beam_length, 500)
    else:
        x = x_ref

    kappa = -F * (beam_length - x) / (E * I_func(x))

    theta = cumtrapz(kappa, x, initial=0)
    y = cumtrapz(theta, x, initial=0)

    return x, y


# ===============================
# 误差分析
# ===============================
def error_analysis(x_rec, y_rec, F):
    x_true_full, y_true_full = theoretical_curve(F, x_rec)
    err = np.abs(y_rec - y_true_full)
    mean_err = np.mean(err)
    max_err = np.max(err)
    rmse = np.sqrt(np.mean(err**2))
    return mean_err, max_err, rmse, x_true_full, y_true_full


# ===============================
# 力估计（加稳定性）
# ===============================
def estimate_force(s, kappa_measured):

    EI = I_func(s) * E
    phi = (beam_length - s) / EI

    numerator = np.sum(kappa_measured * phi)
    denominator = np.sum(phi**2)

    #   防止除0（数值稳定）
    if denominator < 1e-12:
        return 0.0

    F_est = - numerator / denominator

    return F_est


# ===============================
# 理论曲率
# ===============================

def generate_beam_mesh(x_centerline, y_centerline, nx=60, nz=16):
    """
    Generate 3D surface meshes for a variable-section beam.

    Coordinate system for plotting:
        X = beam length (0 → 300)
        Y = width direction  (±b(x)/2)
        Z = bending direction (vertical, displacement)

    Returns:
        X, Y, Z_top, Z_bottom : meshgrids for top/bottom surfaces
        disp_map              : displacement magnitude at each grid point
        x_uni, y_uni, b_uni, theta_uni : centerline data
    """
    x_uni = np.linspace(0, beam_length, nx)
    y_uni = np.interp(x_uni, x_centerline, y_centerline)

    dy = np.gradient(y_uni, x_uni)
    theta = np.arctan2(dy, 1.0)

    b = b_func(x_uni)
    z_norm = np.linspace(-1, 1, nz)

    X = np.tile(x_uni[:, np.newaxis], (1, nz))
    half_w = b / 2.0
    Y = half_w[:, np.newaxis] * z_norm[np.newaxis, :]
    ct = np.cos(theta)
    Z_top = y_uni[:, np.newaxis] + (h / 2.0) * ct[:, np.newaxis]
    Z_bottom = y_uni[:, np.newaxis] - (h / 2.0) * ct[:, np.newaxis]
    disp_map = np.tile(np.abs(y_uni)[:, np.newaxis], (1, nz))

    return X, Y, Z_top, Z_bottom, disp_map, x_uni, y_uni, b, theta


def generate_side_meshes(x_uni, y_uni, b_uni, theta_uni, nside=2):
    """
    Generate left and right side surface meshes (thickness faces).
    Y is constant (±b/2), Z varies from bottom to top.

    Returns (X_left, Y_left, Z_left, X_right, Y_right, Z_right).
    """
    X_side = np.tile(x_uni[:, np.newaxis], (1, nside))
    half_w = b_uni / 2.0
    ct = np.cos(theta_uni)
    z_bottom = y_uni - (h / 2.0) * ct
    z_top = y_uni + (h / 2.0) * ct

    Y_left = np.tile(-half_w[:, np.newaxis], (1, nside))
    Y_right = np.tile(half_w[:, np.newaxis], (1, nside))
    t = np.linspace(0, 1, nside)
    Z_left = z_bottom[:, np.newaxis] + (z_top - z_bottom)[:, np.newaxis] * t[np.newaxis, :]
    Z_right = Z_left.copy()

    return X_side, Y_left, Z_left, X_side, Y_right, Z_right