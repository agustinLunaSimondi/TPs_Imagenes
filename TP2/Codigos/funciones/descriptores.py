import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.contornos import obtener_contornos, visualizar_contornos

def calcular_momentos_hu(contorno, log_value = True):
    momento_hu = cv2.HuMoments(cv2.moments(contorno)).flatten()
    log_momento_hu = -np.sign(momento_hu) * np.log10(np.abs(momento_hu))
    
    if log_value:
        return log_momento_hu
    else:
        return momento_hu
    

def filtrar_contorno_fourier(contornos, idx_contorno, n_descriptor = 50):
    contorno = contornos[idx_contorno]
    contorno_complejo = np.array([point[0][0] + 1j * point[0][1] for point in contorno])
    fourier = np.fft.fft(contorno_complejo)
    fourier_filtrada = np.zeros(fourier.shape, dtype=complex)
    fourier_filtrada[:n_descriptor] = fourier[:n_descriptor]
    fourier_filtrada[-n_descriptor:] = fourier[-n_descriptor:]

    contorno_reconstruido = np.fft.ifft(fourier_filtrada)
    contorno_reconstruido = np.stack((contorno_reconstruido.real, contorno_reconstruido.imag), axis=-1).astype(int)

    return contorno_reconstruido