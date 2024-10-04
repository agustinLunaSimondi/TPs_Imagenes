import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

def filtro_gabor(imagen, ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5, ps=0):
    f_gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, ps, ktype=cv2.CV_32F)
    imagen_filtrada = cv2.filter2D(imagen, cv2.CV_8UC3, f_gabor)
    return imagen_filtrada


def gabor_features(imagen):
    media = np.mean(imagen)
    varianza = np.var(imagen)
    return media, varianza


def transformar_fourier(imagen):
    f_trans = np.fft.fft2(imagen)
    f_shift = np.fft.fftshift(f_trans)
    mag_esp = 20 * np.log(np.abs(f_shift) + 1)
    return f_shift, mag_esp


def fourier_features(f_shift):
    energia = np.sum(np.abs(f_shift))

    h, w = f_shift.shape
    c_h, c_w = h // 2, w // 2
    PerC = f_shift[:c_h, :c_w]
    SdoC = f_shift[:c_h, c_w:]
    TerC = f_shift[c_h:, :c_w]
    CtoC = f_shift[c_h:, c_w:]

    ener_1erC = np.sum(np.abs(PerC))
    ener_2doC = np.sum(np.abs(SdoC))
    ener_3erC = np.sum(np.abs(TerC))
    ener_4toC = np.sum(np.abs(CtoC))

    return energia, ener_1erC, ener_2doC, ener_3erC, ener_4toC


def transformar_wavelet(imagen, wavelet='haar', level=2):
    coef = pywt.wavedec2(imagen, wavelet=wavelet, level=level)
    return coef


def wavelet_features(coef):
    cA = coef[0]
    cHVD = coef[1:]

    energia_approx = np.sum(np.square(cA))

    energia_detalles = []
    for i, (cH, cV, cD) in enumerate(cHVD):
        energia_H = np.sum(np.square(cH))
        energia_V = np.sum(np.square(cV))
        energia_D = np.sum(np.square(cD))
        energia_detalles.append((energia_H, energia_V, energia_D))

    return energia_approx, energia_detalles