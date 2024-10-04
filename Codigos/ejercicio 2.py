import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.generales import plot_1, plot_2
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.descriptores import calcular_momentos_fourier, calcular_momentos_hu
from funciones.contornos import visualizar_contornos, obtener_contornos
from funciones.analizar_texturas import filtro_gabor, gabor_features, transformar_fourier, fourier_features, transformar_wavelet, wavelet_features
from funciones.hough_y_harris import lineas_hough, harris

def ejercicio3():
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'

    imagenes_originales = cargar_imagenes(sufijos_set2, ruta_base_set2)

    imagen_ej = imagenes_originales[2]
    copia_imagen_ej = imagen_ej.copy()
    lineas = lineas_hough(copia_imagen_ej, longitud = 100)
    #print(lineas)

    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen_ej, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Imagen con líneas', imagen_ej)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    ejercicio3()