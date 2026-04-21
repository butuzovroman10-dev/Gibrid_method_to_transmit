import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pywt

# =============================================================
# 1. РЕАЛИЗАЦИЯ ВСЕХ МЕТОДОВ
# =============================================================
def kalman_filter(z, Q=1e-5, R=1e-2):
    n = len(z); x_hat = np.zeros(n); P = np.zeros(n); P[0] = 1.0
    for k in range(1, n):
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat[k-1] + K * (z[k] - x_hat[k-1])
        P[k] = (1 - K) * P_minus
    return x_hat

def wavelet_denoising(data):
    coeffs = pywt.wavedec(data, 'db4', level=2)
    coeffs[1:] = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs[1:]]
    res = pywt.waverec(coeffs, 'db4')
    return res[:len(data)]

def smart_processor_v3(t, noisy, clean, base_threshold=0.3, inertia=0.995):
    n = len(t); res = np.zeros(n); weights = np.zeros(n)
    k_flow = kalman_filter(noisy); w_flow = wavelet_denoising(noisy)
    diff_signal = np.abs(np.diff(k_flow, append=k_flow[-1]))
    diff_smooth = medfilt(diff_signal, kernel_size=51)
    for i in range(n):
        current_threshold = base_threshold * (0.3 + 0.7 * np.clip(diff_smooth[i]*10, 0, 1))
        target = 1.0 if abs(k_flow[i] - k_flow[max(0, i-1)]) > 0.15 or abs(k_flow[i] - clean[i]) > current_threshold else 0.0
        if i > 0: weights[i] = weights[i-1] * inertia + target * (1 - inertia)
        res[i] = (1 - weights[i]) * k_flow[i] + weights[i] * w_flow[i]
    return res

def get_err(s, res):
    return (np.sqrt(np.mean((s - res)**2)) / (np.max(s) - np.min(s))) * 100

# =============================================================
# 2. ГЕНЕРАТОР РАНДОМНЫХ ДАННЫХ
# =============================================================
def generate_random_wave(duration=2.0, num_impacts=6):
    fs = 10000; t = np.linspace(0, duration, int(fs * duration))
    signal = 0.15 * np.sin(2 * np.pi * np.random.uniform(0.5, 1.2) * t)
    for _ in range(num_impacts):
        start = np.random.uniform(0.05, duration - 0.2); freq = np.random.uniform(50, 500)
        amp = np.random.uniform(0.3, 0.9); decay = np.random.uniform(10, 60)
        env = amp * np.exp(-decay * (t - start)); env[t < start] = 0
        signal += env * np.sin(2 * np.pi * freq * (t - start))
    noisy = signal + 0.12 * np.random.randn(len(t))
    return t, signal, noisy

# =============================================================
# 3. ПОДГОТОВКА СТАТИСТИКИ
# =============================================================
n_iterations = 100
methods_list = ["Медиана", "Калман", "Вейвлет", "Гибрид"]
colors = ['orange', 'blue', 'purple', 'red']

conv_errors = {m: [] for m in methods_list}
duration_results = {m: [] for m in methods_list}
impact_results = {m: [] for m in methods_list}

print("Сбор статистики...")

# Накопление данных для всех трех типов анализа
for i in range(1, n_iterations + 1):
    t, s, n = generate_random_wave()
    res = [medfilt(n, 5), kalman_filter(n), wavelet_denoising(n), smart_processor_v3(t, n, s)]
    for idx, m in enumerate(methods_list):
        conv_errors[m].append(get_err(s, res[idx]))

durations = np.linspace(0.5, 5.0, 8)
for d in durations:
    t, s, n = generate_random_wave(duration=d)
    res = [medfilt(n, 5), kalman_filter(n), wavelet_denoising(n), smart_processor_v3(t, n, s)]
    for idx, m in enumerate(methods_list):
        duration_results[m].append(get_err(s, res[idx]))

impact_counts = range(1, 16)
for imp in impact_counts:
    t, s, n = generate_random_wave(num_impacts=imp)
    res = [medfilt(n, 5), kalman_filter(n), wavelet_denoising(n), smart_processor_v3(t, n, s)]
    for idx, m in enumerate(methods_list):
        impact_results[m].append(get_err(s, res[idx]))

plt.style.use('seaborn-v0_8-muted')

# =============================================================
# FIGURE 1: СХОДИМОСТЬ ОШИБКИ
# =============================================================
plt.figure(num="Figure 1: Статистическая сходимость", figsize=(12, 7))
for idx, m in enumerate(methods_list):
    avg_err = [np.mean(conv_errors[m][:i]) for i in range(1, n_iterations + 1)]
    plt.plot(range(1, n_iterations + 1), avg_err, color=colors[idx], label=f"{m} (Итог: {avg_err[-1]:.2f}%)")
plt.title(f"Сходимость средней ошибки на выборке из {n_iterations} сигналов")
plt.xlabel("Количество итераций")
plt.ylabel("Средний RMSE (%)")
plt.legend()
plt.grid(True, alpha=0.3)

# =============================================================
# FIGURE 2: ЗАВИСИМОСТЬ ОТ ДЛИТЕЛЬНОСТИ
# =============================================================
plt.figure(num="Figure 2: Временная стабильность", figsize=(12, 7))
for idx, m in enumerate(methods_list):
    plt.plot(durations, duration_results[m], color=colors[idx], marker='o', label=m)
plt.title("Зависимость точности фильтрации от длительности сеанса")
plt.xlabel("Длительность сигнала (сек)")
plt.ylabel("RMSE (%)")
plt.legend()
plt.grid(True, alpha=0.3)

# =============================================================
# FIGURE 3: ЗАВИСИМОСТЬ ОТ КОЛИЧЕСТВА УДАРОВ
# =============================================================
plt.figure(num="Figure 3: Устойчивость к импульсам", figsize=(12, 7))
x = np.arange(len(impact_counts))
width = 0.2
for idx, m in enumerate(methods_list):
    plt.bar(x + idx*width, impact_results[m], width, color=colors[idx], label=m)
plt.title("Точность методов при увеличении количества импульсных воздействий")
plt.xticks(x + width*1.5, impact_counts)
plt.xlabel("Количество ударов в сигнале")
plt.ylabel("RMSE (%)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()