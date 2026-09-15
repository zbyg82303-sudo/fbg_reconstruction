import numpy as np

sensor_positions = np.array([60, 120, 180, 240])
_LS_SOLVERS = {}


def _ls_solver(n_top):
    solver = _LS_SOLVERS.get(n_top)
    if solver is not None:
        return solver

    s = sensor_positions[:n_top]
    A = np.vstack([s**2, s, np.ones_like(s)]).T
    W = np.eye(n_top)
    if n_top > 1:
        W[1, 1] = 4
    W_sqrt = np.sqrt(W)
    solver = np.linalg.pinv(W_sqrt @ A) @ W_sqrt
    _LS_SOLVERS[n_top] = solver
    return solver

def curvature_least_squares(strain, sensor_y=1.75):
    n_top = min(len(strain), 4)
    strain_top = strain[:n_top]
    kappa = strain_top / sensor_y
    return _ls_solver(n_top) @ kappa


def curvature_reconstruction(s, coef):
    a, b, c = coef
    return a * s**2 + b * s + c


def frenet_reconstruction_arc(s, kappa):
    """Frenet 递推积分 (向量化): 逐段累加的切向角用 cumsum 一次求得, 直线段与
    曲率段分别成组计算后再合并, 结果与逐点循环完全一致."""
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    n = s.size
    if n < 2:
        return np.zeros(n), np.zeros(n), np.zeros(n)

    ds = np.diff(s)
    k_avg = 0.5 * (kappa[1:] + kappa[:-1])
    dtheta = k_avg * ds

    # 每段起点的切向角 (theta[0] = 0, 之后逐段累加, 与原循环一致)
    theta = np.empty(ds.size)
    theta[0] = 0.0
    np.cumsum(dtheta[:-1], out=theta[1:])

    straight = np.abs(k_avg) < 1e-8
    sin_dt = np.sin(dtheta)
    cos_dt = np.cos(dtheta)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    inv_k = 1.0 / np.where(straight, 1.0, k_avg)   # 直线段占位, 避免除零
    dx = np.where(straight,
                  cos_th * ds,
                  (cos_th * sin_dt - sin_th * (1.0 - cos_dt)) * inv_k)
    dy = np.where(straight,
                  sin_th * ds,
                  (sin_th * sin_dt + cos_th * (1.0 - cos_dt)) * inv_k)

    x = np.zeros(n)
    y = np.zeros(n)
    np.cumsum(dx, out=x[1:])
    np.cumsum(dy, out=y[1:])

    return x, y, np.zeros(n)
