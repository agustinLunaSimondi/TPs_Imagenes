import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.generales import plot_1, plot_2
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.descriptores import filtrar_contorno_fourier, calcular_momentos_hu
from funciones.contornos import visualizar_contornos, obtener_contornos, segmentar_contorno
from funciones.analizar_texturas import filtro_gabor, gabor_features, transformar_fourier, fourier_features, transformar_wavelet, wavelet_features
from funciones.hough_y_harris import lineas_hough, harris

def ejercicio2b():
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'

    imagenes_originales = cargar_imagenes(sufijos_set2, ruta_base_set2)
    imagen_1 = imagenes_originales[0]
    imagen_2 = imagenes_originales[1]
    imagen_5 = imagenes_originales[4]

    if len(imagen_1.shape) == 3:
        imagen_1 = cv2.cvtColor(imagen_1, cv2.COLOR_BGR2GRAY)
    if len(imagen_2.shape) == 3:
        imagen_2 = cv2.cvtColor(imagen_2, cv2.COLOR_BGR2GRAY)
    if len(imagen_5.shape) == 3:
        imagen_5 = cv2.cvtColor(imagen_5, cv2.COLOR_BGR2GRAY)

    contornos_1 = obtener_contornos(imagen_1, 170)
    contorno_1 = contornos_1[240]
    contornos_2 = obtener_contornos(imagen_2, 100)
    contorno_2 = contornos_2[74]
    contornos_5 = obtener_contornos(imagen_5, 100)
    contorno_5 = contornos_5[137]

    imagen_contorno_1 = segmentar_contorno(imagen_1, contorno_1)
    imagen_contorno_2 = segmentar_contorno(imagen_2, contorno_2)
    imagen_contorno_5 = segmentar_contorno(imagen_5, contorno_5)

    # gabor
    orientaciones = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frecuencias = [5, 10, 15]
    features_1 = []
    features_2 = []
    features_5 = []
    
    for theta in orientaciones:
        for lambd in frecuencias:
            imagen_gabor_1 = filtro_gabor(imagen_contorno_1, theta=theta, lambd=lambd)
            imagen_gabor_2 = filtro_gabor(imagen_contorno_2, theta=theta, lambd=lambd)
            imagen_gabor_5 = filtro_gabor(imagen_contorno_5, theta=theta, lambd=lambd)
            m_val_1, var_val_1 = gabor_features(imagen_gabor_1)
            m_val_2, var_val_2 = gabor_features(imagen_gabor_2)
            m_val_5, var_val_5 = gabor_features(imagen_gabor_5)
            features_1.append((theta, lambd, m_val_1, var_val_1))
            features_2.append((theta, lambd, m_val_2, var_val_2))
            features_5.append((theta, lambd, m_val_5, var_val_5))
    
    print("\nCaracterísticas de Gabor img-1 (theta, lambda, media, varianza):")
    for features in features_1:
        print(f"Theta: {features[0]:.2f} rad, Frecuencia: {features[1]:.2f}, Media: {features[2]:.2f}, Varianza: {features[3]:.2f}")
    print("\nCaracterísticas de Gabor img-2 (theta, lambda, media, varianza):")
    for features in features_2:
        print(f"Theta: {features[0]:.2f} rad, Frecuencia: {features[1]:.2f}, Media: {features[2]:.2f}, Varianza: {features[3]:.2f}")
    print("\nCaracterísticas de Gabor img-5 (theta, lambda, media, varianza):")
    for features in features_5:
        print(f"Theta: {features[0]:.2f} rad, Frecuencia: {features[1]:.2f}, Media: {features[2]:.2f}, Varianza: {features[3]:.2f}")

    fig, axs = plt.subplots(len(orientaciones), len(frecuencias), figsize=(12, 12))
    for i, theta in enumerate(orientaciones):
        for j, lambd in enumerate(frecuencias):
            imagen_filtrada = filtro_gabor(imagen_contorno_1, theta=theta, lambd=lambd)
            axs[i, j].imshow(imagen_filtrada, cmap='gray')
            axs[i, j].set_title(f'θ={theta:.2f}, λ={lambd}')
            axs[i, j].axis('off')
    plt.show()

    fig, axs = plt.subplots(len(orientaciones), len(frecuencias), figsize=(12, 12))
    for i, theta in enumerate(orientaciones):
        for j, lambd in enumerate(frecuencias):
            imagen_filtrada = filtro_gabor(imagen_contorno_2, theta=theta, lambd=lambd)
            axs[i, j].imshow(imagen_filtrada, cmap='gray')
            axs[i, j].set_title(f'θ={theta:.2f}, λ={lambd}')
            axs[i, j].axis('off')
    plt.show()

    fig, axs = plt.subplots(len(orientaciones), len(frecuencias), figsize=(12, 12))
    for i, theta in enumerate(orientaciones):
        for j, lambd in enumerate(frecuencias):
            imagen_filtrada = filtro_gabor(imagen_contorno_5, theta=theta, lambd=lambd)
            axs[i, j].imshow(imagen_filtrada, cmap='gray')
            axs[i, j].set_title(f'θ={theta:.2f}, λ={lambd}')
            axs[i, j].axis('off')
    plt.show()

    # fourier
    f_shift_1, mag_esp_1 = transformar_fourier(imagen_contorno_1)
    energia_total_1, energia1_1, energia2_1, energia3_1, energia4_1 = fourier_features(f_shift_1)
    f_shift_2, mag_esp_2 = transformar_fourier(imagen_contorno_2)
    energia_total_2, energia1_2, energia2_2, energia3_2, energia4_2 = fourier_features(f_shift_2)
    f_shift_5, mag_esp_5 = transformar_fourier(imagen_contorno_5)
    energia_total_5, energia1_5, energia2_5, energia3_5, energia4_5 = fourier_features(f_shift_5)

    print("\nCaracterísticas de fourier img-1")
    print(f"Energía total: {energia_total_1:.2f}")
    print(f"Energía cuadrante 1: {energia1_1:.2f}")
    print(f"Energía cuadrante 2: {energia2_1:.2f}")
    print(f"Energía cuadrante 3: {energia3_1:.2f}")
    print(f"Energía cuadrante 4: {energia4_1:.2f}")

    print("\nCaracterísticas de fourier img-2")
    print(f"Energía total: {energia_total_2:.2f}")
    print(f"Energía cuadrante 1: {energia1_2:.2f}")
    print(f"Energía cuadrante 2: {energia2_2:.2f}")
    print(f"Energía cuadrante 3: {energia3_2:.2f}")
    print(f"Energía cuadrante 4: {energia4_2:.2f}")

    print("\nCaracterísticas de fourier img-5")
    print(f"Energía total: {energia_total_5:.2f}")
    print(f"Energía cuadrante 1: {energia1_5:.2f}")
    print(f"Energía cuadrante 2: {energia2_5:.2f}")
    print(f"Energía cuadrante 3: {energia3_5:.2f}")
    print(f"Energía cuadrante 4: {energia4_5:.2f}")

    plot_2(imagen_contorno_1, mag_esp_1, "Contorno seleccionado de img-1", "Espectro de magnitud de fourier")
    plot_2(imagen_contorno_2, mag_esp_2, "Contorno seleccionado de img-2", "Espectro de magnitud de fourier")
    plot_2(imagen_contorno_5, mag_esp_5, "Contorno seleccionado de img-5", "Espectro de magnitud de fourier")
    """

    # wavelets
    """
    wavelets_1 = transformar_wavelet(imagen_contorno_1)
    wavelets_2 = transformar_wavelet(imagen_contorno_2)
    wavelets_5 = transformar_wavelet(imagen_contorno_5)
    energia_approx_1, energia_detalles_1 = wavelet_features(wavelets_1)
    energia_approx_2, energia_detalles_2 = wavelet_features(wavelets_2)
    energia_approx_5, energia_detalles_5 = wavelet_features(wavelets_5)

    print("\nCaracterísticas de wavelets img-1")
    print(f"\nEnergía de la Aproximación (baja frecuencia): {energia_approx_1:.2f}")
    for i, (energia_H, energia_V, energia_D) in enumerate(energia_detalles_1):
        print(f"Energía Nivel {i+1} - Horizontal: {energia_H:.2f}, Vertical: {energia_V:.2f}, Diagonal: {energia_D:.2f}")
    print("\nCaracterísticas de wavelets img-2")
    print(f"\nEnergía de la Aproximación (baja frecuencia): {energia_approx_2:.2f}")
    for i, (energia_H, energia_V, energia_D) in enumerate(energia_detalles_2):
        print(f"Energía Nivel {i+1} - Horizontal: {energia_H:.2f}, Vertical: {energia_V:.2f}, Diagonal: {energia_D:.2f}")
    print("\nCaracterísticas de wavelets img-5")
    print(f"\nEnergía de la Aproximación (baja frecuencia): {energia_approx_5:.2f}")
    for i, (energia_H, energia_V, energia_D) in enumerate(energia_detalles_5):
        print(f"Energía Nivel {i+1} - Horizontal: {energia_H:.2f}, Vertical: {energia_V:.2f}, Diagonal: {energia_D:.2f}")

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(imagen_contorno_1, cmap='gray')
    axs[0, 0].set_title('Contorno seleccionado de img-1')
    axs[0, 1].imshow(wavelets_1[0], cmap='gray')
    axs[0, 1].set_title('Aproximación (baja frecuencia)')
    cH1, cV1, cD1 = wavelets_1[1]
    axs[1, 0].imshow(cH1, cmap='gray')
    axs[1, 0].set_title('Detalle Horizontal Nivel 1')
    axs[1, 1].imshow(cV1, cmap='gray')
    axs[1, 1].set_title('Detalle Vertical Nivel 1')
    axs[2, 0].imshow(cD1, cmap='gray')
    axs[2, 0].set_title('Detalle Diagonal 1')
    for i in range(3):
        for j in range(2):
            axs[i, j].axis('off')
    plt.axis('off')
    plt.show()

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(imagen_contorno_2, cmap='gray')
    axs[0, 0].set_title('Contorno seleccionado de img-2')
    axs[0, 1].imshow(wavelets_2[0], cmap='gray')
    axs[0, 1].set_title('Aproximación (baja frecuencia)')
    cH1, cV1, cD1 = wavelets_2[1]
    axs[1, 0].imshow(cH1, cmap='gray')
    axs[1, 0].set_title('Detalle Horizontal Nivel 1')
    axs[1, 1].imshow(cV1, cmap='gray')
    axs[1, 1].set_title('Detalle Vertical Nivel 1')
    axs[2, 0].imshow(cD1, cmap='gray')
    axs[2, 0].set_title('Detalle Diagonal 1')
    for i in range(3):
        for j in range(2):
            axs[i, j].axis('off')
    plt.axis('off')
    plt.show()

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs[0, 0].imshow(imagen_contorno_5, cmap='gray')
    axs[0, 0].set_title('Contorno seleccionado de img-5')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(wavelets_5[0], cmap='gray')
    axs[0, 1].set_title('Aproximación (baja frecuencia)')
    cH1, cV1, cD1 = wavelets_5[1]
    axs[1, 0].imshow(cH1, cmap='gray')
    axs[1, 0].set_title('Detalle Horizontal Nivel 1')
    axs[1, 1].imshow(cV1, cmap='gray')
    axs[1, 1].set_title('Detalle Vertical Nivel 1')
    axs[2, 0].imshow(cD1, cmap='gray')
    axs[2, 0].set_title('Detalle Diagonal 1')
    for i in range(3):
        for j in range(2):
            axs[i, j].axis('off')
    plt.axis('off')
    plt.show()

    return


if __name__ == "__main__":
    ejercicio2b()