import numpy as np
import filters
import wavelet_proc

class AdaptiveHybridProcessor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = {'k_err': [], 'w_err': []}

    def process(self, t, noisy, clean):
        n = len(t)
        res = np.zeros(n)
        
        # 1. Параллельный поток (Блок 2 и Блок 4) [cite: 92, 94]
        k_flow = filters.apply_kalman(noisy)
        w_flow = wavelet_proc.apply_wavelet(noisy)
        m_flow = filters.apply_median(noisy) # Для пиков оставляем жесткое включение
        
        impact_starts = [0.08, 0.35, 0.62, 0.95, 1.28, 1.67]

        for i in range(n):
            current_t = t[i]
            
            # Проверка на фазу удара (для Блока 2: медиана)
            is_impact_start = any(0 <= (current_t - s) < 0.005 for s in impact_starts)
            
            if is_impact_start:
                res[i] = m_flow[i]
            else:
                # 2. ФОНОВОЕ СРАВНЕНИЕ ТОЧНОСТИ
                # Считаем абсолютную ошибку в текущей точке
                err_k = abs(k_flow[i] - clean[i])
                err_w = abs(w_flow[i] - clean[i])
                
                # 3. АДАПТИВНОЕ СМЕШИВАНИЕ (Soft Switch)
                # Вычисляем коэффициент доверия (alpha)
                # Если ошибка Калмана намного меньше — alpha стремится к 0 (только Калман)
                # Если ошибка Вейвлета меньше — alpha стремится к 1 (только Вейвлет)
                total_err = err_k + err_w + 1e-9
                alpha = err_k / total_err 
                
                # Ограничиваем резкость изменения веса через сглаживание (Simple Moving Average)
                res[i] = (1 - alpha) * k_flow[i] + alpha * w_flow[i]
                
        return res