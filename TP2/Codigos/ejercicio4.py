import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.generales import plot_1, plot_2
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.descriptores import filtrar_contorno_fourier, calcular_momentos_hu
from funciones.contornos import visualizar_contornos, obtener_contornos
from funciones.analizar_texturas import filtro_gabor, gabor_features, transformar_fourier, fourier_features, transformar_wavelet, wavelet_features
from funciones.hough_y_harris import lineas_hough, harris

def ejercicio4():
    # Hough
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'
    imagenes_originales = cargar_imagenes(sufijos_set2, ruta_base_set2)
    imagen_3 = imagenes_originales[2]
    imagen_4 = imagenes_originales[3]
    copia_imagen_3 = imagen_3.copy()
    copia_imagen_4 = imagen_4.copy()

    lineas_3 = lineas_hough(copia_imagen_3, lower_thersh = 80, upper_thresh = 100, longitud = 150)
    lineas_4 = lineas_hough(copia_imagen_4, lower_thersh = 80, upper_thresh = 100, longitud = 40)
    for linea in lineas_3:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_3, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for linea in lineas_4:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_4, (x1, y1), (x2, y2), (0, 0, 255), 15)

    #plot_2(imagen_3, imagen_4, "Líneas de img-3", "Líneas de img-4")

    # Harris
    imagen_esquinas_3 = harris(copia_imagen_3, k = 0.04, threshold = 85)
    imagen_esquinas_4 = harris(copia_imagen_4, k = 0.04, threshold = 165)
    esquinas_3 = np.argwhere(imagen_esquinas_3 == 1)
    esquinas_4 = np.argwhere(imagen_esquinas_4 == 1)

    for esquina in esquinas_3:
        y, x = esquina
        cv2.circle(imagen_3, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
    for esquina in esquinas_4:
        y, x = esquina
        cv2.circle(imagen_4, (x, y), radius=9, color=(0, 255, 0), thickness=-1)

    #plot_2(imagen_3, imagen_4, "Esquinas de img_3", "Esquinas de img-4")

    plot_2(imagen_3, imagen_4, "Líneas y esquinas de img-3", "Líneas y esquinas de img-4")

    return


if __name__ == "__main__":
    ejercicio4()