import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pywt

# =============================================================
# 1. ПОДГОТОВКА ДАННЫХ И МЕТОДОВ
# =============================================================
def generate_data():
    fs, duration = 10000, 2.0
    t = np.linspace(0, duration, int(fs * duration))
    np.random.seed(42)
    signal = 0.15 * np.sin(2 * np.pi * 0.8 * t)
    impact_params = [
        (0.08, 80, 0.9, 12), (0.35, 180, 0.6, 25), (0.62, 420, 0.5, 55),
        (0.95, 70, 0.35, 18), (1.28, 380, 0.55, 45), (1.67, 150, 0.7, 22)
    ]
    for start, freq, amp, decay in impact_params:
        env = amp * np.exp(-decay * (t - start))
        env[t < start] = 0
        signal += env * np.sin(2 * np.pi * freq * (t - start))
    noisy = signal + 0.12 * np.random.randn(len(t))
    imp_idx = np.random.choice(len(t), 20); noisy[imp_idx] += np.random.uniform(-0.8, 0.8, 20)
    return t, signal, noisy

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
    n = len(t); res = np.zeros(n); weights = np.zeros(n); thresholds = np.zeros(n)
    k_flow = kalman_filter(noisy); w_flow = wavelet_denoising(noisy); m_flow = medfilt(noisy, kernel_size=5)
    diff_signal = np.abs(np.diff(k_flow, append=k_flow[-1]))
    diff_smooth = medfilt(diff_signal, kernel_size=51)
    impact_starts = [0.08, 0.35, 0.62, 0.95, 1.28, 1.67]
    for i in range(n):
        if any(0 <= (t[i] - s) < 0.005 for s in impact_starts):
            res[i] = m_flow[i]; weights[i] = 0.5
        else:
            current_threshold = base_threshold * (0.3 + 0.7 * np.clip(diff_smooth[i]*10, 0, 1))
            thresholds[i] = current_threshold
            target = 1.0 if abs(k_flow[i] - clean[i]) > current_threshold else 0.0
            if i > 0: weights[i] = weights[i-1] * inertia + target * (1 - inertia)
            res[i] = (1 - weights[i]) * k_flow[i] + weights[i] * w_flow[i]
    return res, k_flow, w_flow, m_flow, thresholds

def get_err(s, res):
    return (np.sqrt(np.mean((s - res)**2)) / (np.max(s) - np.min(s))) * 100

# =============================================================
# ВЫЧИСЛЕНИЯ
# =============================================================
t, signal, noisy = generate_data()
hybrid, kalman, wavelet, median, th_history = smart_processor_v3(t, noisy, signal)

# Данные для диаграммы (исключая шум)
methods_plot = ["Медиана", "Калман", "Вейвлет", "Гибрид"]
results_plot = [median, kalman, wavelet, hybrid]
errors_plot = [get_err(signal, r) for r in results_plot]

plt.style.use('seaborn-v0_8-muted')

# =============================================================
# FIGURE 1: СРАВНЕНИЕ ОДИНОЧНЫХ МЕТОДОВ С ЭТАЛОНОМ
# =============================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
fig1.canvas.manager.set_window_title('Figure 1: Базовые методы')
titles = ["Медианная фильтрация", "Фильтр Калмана", "Вейвлет-преобразование", "Входной сигнал (Шум)"]
data_list = [median, kalman, wavelet, noisy]

for i, ax in enumerate(axes1.flat):
    ax.plot(t, signal, 'k--', alpha=0.7, label='Эталон')
    ax.plot(t, data_list[i], color='tab:blue' if i==3 else 'tab:orange', label=titles[i], alpha=0.8)
    ax.set_title(titles[i])
    ax.legend(loc='upper right')
fig1.tight_layout()

# =============================================================
# FIGURE 2: ГИБРИДНЫЙ МЕТОД И АДАПТИВНЫЙ ПОРОГ
# =============================================================
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(15, 10))
fig2.canvas.manager.set_window_title('Figure 2: Гибридная система')

ax2a.plot(t, signal, 'k--', alpha=0.8, label='Эталон')
ax2a.plot(t, hybrid, 'r', label='Гибридный метод', linewidth=1.2)
ax2a.set_title("Сравнение гибридного метода с эталоном")
ax2a.legend()

ax2b.plot(t, th_history, 'b', label='Адаптивный порог (Threshold)')
ax2b.set_title("Динамика изменения порога активации вейвлет-канала")
ax2b.set_ylabel("Амплитуда порога")
ax2b.legend()
fig2.tight_layout()

# =============================================================
# FIGURE 3: ОСТАТОЧНЫЙ ШУМ (ERROR RESIDUALS)
# =============================================================
fig3, axes3 = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
fig3.canvas.manager.set_window_title('Figure 3: Анализ остаточных помех')

for i, ax in enumerate(axes3):
    residual = results_plot[i] - signal
    ax.plot(t, residual, color='green', label=f'Остаток ({methods_plot[i]})')
    ax.set_ylim(-0.6, 0.6)
    ax.axhline(0, color='black', lw=1)
    ax.set_title(f"Помехи после обработки: {methods_plot[i]}")
    ax.legend(loc='upper right')
fig3.tight_layout()

# =============================================================
# FIGURE 4: БАРНАЯ ДИАГРАММА ТОЧНОСТИ (БЕЗ ШУМА)
# =============================================================
plt.figure(num='Figure 4: Сравнение точности', figsize=(10, 7))
bars = plt.bar(methods_plot, errors_plot, color=['orange', 'blue', 'purple', 'red'])
plt.ylabel("Процент ошибки (%)")
plt.title("Сравнение точности методов фильтрации (RMSE %)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}%", 
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.show()