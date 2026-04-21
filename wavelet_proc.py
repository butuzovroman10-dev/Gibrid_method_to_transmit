import pywt

def apply_wavelet(data): # 
    coeffs = pywt.wavedec(data, 'db4', level=2)
    coeffs[1:] = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'db4')[:len(data)]