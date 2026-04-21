import numpy as np

def generate_telemetry():
    fs, duration = 10000, 2.0
    t = np.linspace(0, duration, int(fs * duration))
    np.random.seed(42)
    
    # 1. Полезный сигнал s(t)
    # Медленный дрейф (низкочастотный процесс)
    slow_drift = 0.15 * np.sin(2 * np.pi * 0.8 * t)
    
    def impact(t, start, freq, amp, decay):
        res = amp * np.exp(-decay * (t - start)) * np.sin(2 * np.pi * freq * (t - start))
        res[t < start] = 0
        return res

    # Все 6 ударов с разными частотами и затуханием
    signal = slow_drift.copy()
    impacts = [
        (0.08, 80, 0.9, 12), (0.35, 180, 0.6, 25), (0.62, 420, 0.5, 55),
        (0.95, 70, 0.35, 18), (1.28, 380, 0.55, 45), (1.67, 150, 0.7, 22)
    ]
    for p in impacts:
        signal += impact(t, *p)
    
    # 2. Шум n(t) и импульсные помехи
    white_noise = 0.12 * np.random.randn(len(t))
    noisy = signal + white_noise
    imp_idx = np.random.choice(len(t), 15)
    noisy[imp_idx] += np.random.uniform(-0.7, 0.7, 15)
    
    return t, signal, noisy