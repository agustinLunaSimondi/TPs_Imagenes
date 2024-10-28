import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform, img_as_float
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from scipy.optimize import differential_evolution
from scipy import fftpack
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.generales import plot_1, plot_2
from funciones.MI import registrar_MI, fondo_blanco_a_negro, invertir
from funciones.CNN import registrar_cnn

def ejercicio4():

    sufijos = ['Fimg-11', 'img-1', 'img-2', 'img-12','img-21','img-22']
    ruta_base = './TP3/Imagenes/PAIByB-7/'

    imagenes = cargar_imagenes(sufijos, ruta_base)
    #mostrar_imagenes(imagenes, sufijos)

    imagen_1 = imagenes[1]
    imagen_2 = imagenes[2]
    imagen_11 = imagenes[0]
    imagen_12 = imagenes[3]
    imagen_21 = imagenes[4]
    imagen_22 = imagenes[5]

    #imagen_2 = fondo_blanco_a_negro(imagen_2, I_threshold = 250)
    #imagen_12 = fondo_blanco_a_negro(imagen_12, I_threshold = 250)
    #imagen_21 = fondo_blanco_a_negro(imagen_21, I_threshold = 250)

    invertida_2 = invertir(imagen_2)
    invertida_12 = invertir(imagen_12)
    invertida_21 = invertir(imagen_21)
    imagen_21_fn = fondo_blanco_a_negro(imagen_21, 240)

    imagen_1 = img_as_float(imagen_1)
    imagen_2 = img_as_float(imagen_2)
    imagen_11 = img_as_float(imagen_11)
    imagen_12 = img_as_float(imagen_12)
    imagen_21 = img_as_float(imagen_21)
    imagen_22 = img_as_float(imagen_22)

    decena = 2
    permutar = False

    if decena == 0:
        if not permutar:
            registrar_MI(imagen_1, invertida_2)
        else:
            registrar_MI(invertida_2, imagen_1)
    elif decena == 1:
        if not permutar:
            registrar_MI(imagen_11, invertida_12)
        else:
            registrar_MI(invertida_12, imagen_11)
    elif decena == 2:
        if not permutar:
            registrar_MI(imagen_21_fn, imagen_22)
        else:
            registrar_MI(imagen_22, imagen_21_fn)



    

    #registrar_cnn(imagen_1, invertida_2)
    #registrar_cnn(imagen_11, invertida_12)
    #registrar_cnn(imagen_21_fn, imagen_22)
            
    return

def prueba():
    sufijos = ['Fimg-11', 'img-1', 'img-2', 'img-12','img-21','img-22']
    ruta_base = './TP3/Imagenes/PAIByB-7/'

    imagenes = cargar_imagenes(sufijos, ruta_base)
    imagen_1 = imagenes[1]
    imagen_2 = imagenes[2]
    imagen_11 = imagenes[0]
    imagen_12 = imagenes[3]
    imagen_21 = imagenes[4]
    imagen_22 = imagenes[5]

    imagen_2_n = fondo_blanco_a_negro(imagen_2, I_threshold = 250)
    imagen_12_n = fondo_blanco_a_negro(imagen_12, I_threshold = 250)
    imagen_21_n = fondo_blanco_a_negro(imagen_21, I_threshold = 250)

    plot_2(imagen_2, imagen_2_n)
    plot_2(imagen_12, imagen_12_n)
    plot_2(imagen_21, imagen_21_n)

    return


if __name__ == "__main__":
    ejercicio4()