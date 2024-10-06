import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.generales import plot_1, plot_2
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.descriptores import filtrar_contorno_fourier, calcular_momentos_hu
from funciones.contornos import visualizar_contornos, obtener_contornos
from funciones.analizar_texturas import filtro_gabor, gabor_features, transformar_fourier, fourier_features, transformar_wavelet, wavelet_features
from funciones.hough_y_harris import lineas_hough, harris

def ejercicio2a():
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'

    imagenes_originales = cargar_imagenes(sufijos_set2, ruta_base_set2)

    # img-1
    imagen_1 = imagenes_originales[0]
    contornos_1 = obtener_contornos(imagen_1, 170)
    contorno_fourier_1 = filtrar_contorno_fourier(contornos_1, 240)
    imagen_contorno_1 = imagen_1.copy()
    imagen_contorno_fourier_1 = imagen_1.copy()
    cv2.drawContours(imagen_contorno_1, [contornos_1[240]], -1, (0, 0, 255), 2)
    for point in contorno_fourier_1:
        cv2.circle(imagen_contorno_fourier_1, (point[0], point[1]), 1, (0, 0, 255), -1)

    plot_2(imagen_contorno_1, imagen_contorno_fourier_1, "Contorno original img-1", "Contorno reconstruido img-1")
    mo_hu_1 = calcular_momentos_hu(contornos_1[240])
    mo_hu_fourier_1 = calcular_momentos_hu(contorno_fourier_1)
    print("\nMomentos Hu img-1:")
    for i, momento in enumerate(mo_hu_1):
        print(f"Momento Hu {i+1}: {momento}")
    print("\nMomentos Hu fourier img-1:")
    for i, momento in enumerate(mo_hu_fourier_1):
        print(f"Momento Hu {i+1}: {momento}")
    
    # img-2
    imagen_2 = imagenes_originales[1]
    contornos_2 = obtener_contornos(imagen_2, 100)
    contorno_fourier_2 = filtrar_contorno_fourier(contornos_2, 74)
    imagen_contorno_2 = imagen_2.copy()
    imagen_contorno_fourier_2 = imagen_2.copy()
    cv2.drawContours(imagen_contorno_2, [contornos_2[74]], -1, (0, 0, 255), 2)
    for point in contorno_fourier_2:
        cv2.circle(imagen_contorno_fourier_2, (point[0], point[1]), 1, (0, 0, 255), -1)

    plot_2(imagen_contorno_2, imagen_contorno_fourier_2, "Contorno original img-2", "Contorno reconstruido img-2")
    mo_hu_2 = calcular_momentos_hu(contornos_2[74])
    mo_hu_fourier_2 = calcular_momentos_hu(contorno_fourier_2)
    print("\nMomentos Hu img-2:")
    for i, momento in enumerate(mo_hu_2):
        print(f"Momento Hu {i+1}: {momento}")
    print("\nMomentos Hu fourier img-2:")
    for i, momento in enumerate(mo_hu_fourier_2):
        print(f"Momento Hu {i+1}: {momento}")

    # img-5
    imagen_5 = imagenes_originales[4]
    contornos_5 = obtener_contornos(imagen_5, 100)
    contorno_fourier_5 = filtrar_contorno_fourier(contornos_5, 137)
    imagen_contorno_5 = imagen_5.copy()
    imagen_contorno_fourier_5 = imagen_5.copy()
    cv2.drawContours(imagen_contorno_5, [contornos_5[137]], -1, (0, 0, 255), 2)
    for point in contorno_fourier_5:
        cv2.circle(imagen_contorno_fourier_5, (point[0], point[1]), 1, (0, 0, 255), -1)

    plot_2(imagen_contorno_5, imagen_contorno_fourier_5, "Contorno original img-5", "Contorno reconstruido img-5")
    mo_hu_5 = calcular_momentos_hu(contornos_5[137])
    mo_hu_fourier_5 = calcular_momentos_hu(contorno_fourier_5)
    print("\nMomentos Hu img-5:")
    for i, momento in enumerate(mo_hu_5):
        print(f"Momento Hu {i+1}: {momento}")
    print("\nMomentos Hu fourier img-5:")
    for i, momento in enumerate(mo_hu_fourier_5):
        print(f"Momento Hu {i+1}: {momento}")

    return


if __name__ == "__main__":
    ejercicio2a()