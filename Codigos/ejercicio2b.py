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

    contornos_1 = obtener_contornos(imagen_1, 170)
    contorno_1 = contornos_1[240]
    contornos_2 = obtener_contornos(imagen_2, 100)
    contorno_2 = contornos_1[74]
    contornos_5 = obtener_contornos(imagen_5, 100)
    contorno_5 = contornos_1[137]

    imagen_contorno_1 = segmentar_contorno(imagen_1, contorno_1)
    imagen_contorno_2 = segmentar_contorno(imagen_2, contorno_2)
    imagen_contorno_5 = segmentar_contorno(imagen_5, contorno_5)



    return


if __name__ == "__main__":
    ejercicio2b()