import numpy as np
from scipy.signal import medfilt

def apply_kalman(z, Q=1e-5, R=1e-2): # [cite: 46, 49]
    n = len(z)
    x_hat, P = np.zeros(n), np.zeros(n)
    P[0] = 1.0
    for k in range(1, n):
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat[k-1] + K * (z[k] - x_hat[k-1])
        P[k] = (1 - K) * P_minus
    return x_hat

def apply_median(z, size=5): # [cite: 52, 54]
    return medfilt(z, kernel_size=size)