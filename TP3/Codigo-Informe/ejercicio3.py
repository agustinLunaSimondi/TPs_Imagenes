import numpy as np
import matplotlib.pyplot as plt
import cv2
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.harris import harris_kp, registrar
from funciones.generales import plot_1, plot_2

def ejercicio3():

    sufijos = ['img-1', 'img-2', 'img-3', 'img-4','img-11','img-12']
    ruta_base = './TP3/Imagenes/PAIByB-5/'

    imagenes = cargar_imagenes(sufijos, ruta_base)
 
    imagen_1 = imagenes[0]
    imagen_2 = imagenes[1]
    imagen_3 = imagenes[2]
    imagen_4 = imagenes[3]
    imagen_11 = imagenes[4]
    imagen_12 = imagenes[5]

    copia_imagen_1 = imagen_1.copy()
    copia_imagen_2 = imagen_2.copy()
    copia_imagen_3 = imagen_3.copy()
    copia_imagen_4 = imagen_4.copy()
    copia_imagen_11 = imagen_11.copy()
    copia_imagen_12 = imagen_12.copy()

    img_movil = 12
    decena = img_movil//10
    sift = cv2.SIFT_create()

    if decena == 0:
        kp_1, imagen_esquinas_1 = harris_kp(copia_imagen_1, threshold = 70) # 77 para la 4
        esquinas_1 = np.argwhere(imagen_esquinas_1 == 1)
        for esquina in esquinas_1:
            y, x = esquina
            cv2.circle(imagen_1, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
        _, descriptores_1 = sift.compute(imagen_1, kp_1)
        #imagen_1 = cv2.drawKeypoints(imagen_1, kp_1, imagen_1)

        if img_movil == 2:
            kp_2, imagen_esquinas_2 = harris_kp(copia_imagen_2, threshold = 70)
            esquinas_2 = np.argwhere(imagen_esquinas_2 == 1)
            for esquina in esquinas_2:
                y, x = esquina
                cv2.circle(imagen_2, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
            _, descriptores_2 = sift.compute(imagen_2, kp_2)
            #imagen_2 = cv2.drawKeypoints(imagen_2, kp_2, imagen_2)

            registrar(imagen_1, imagen_2, kp_1, kp_2, descriptores_1, descriptores_2)

        elif img_movil == 3:
            kp_3, imagen_esquinas_3 = harris_kp(copia_imagen_3, threshold = 120)
            esquinas_3 = np.argwhere(imagen_esquinas_3 == 1)
            for esquina in esquinas_3:
                y, x = esquina
                cv2.circle(imagen_3, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
            _, descriptores_3 = sift.compute(imagen_3, kp_3)
            #imagen_3 = cv2.drawKeypoints(imagen_3, kp_3, imagen_3)

            registrar(imagen_1, imagen_3, kp_1, kp_3, descriptores_1, descriptores_3) 

        elif img_movil == 4:
            kp_4, imagen_esquinas_4 = harris_kp(copia_imagen_4, threshold = 90)
            esquinas_4 = np.argwhere(imagen_esquinas_4 == 1)
            for esquina in esquinas_4:
                y, x = esquina
                cv2.circle(imagen_4, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
            _, descriptores_4 = sift.compute(imagen_4, kp_4)
            #imagen_4 = cv2.drawKeypoints(imagen_4, kp_4, imagen_4)

            registrar(imagen_1, imagen_4, kp_1, kp_4, descriptores_1, descriptores_4)

    elif decena == 1:
        kp_11, imagen_esquinas_11 = harris_kp(copia_imagen_11, threshold = 40)
        esquinas_11 = np.argwhere(imagen_esquinas_11 == 1)
        for esquina in esquinas_11:
            y, x = esquina
            cv2.circle(imagen_11, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
        _, descriptores_11 = sift.compute(imagen_11, kp_11)
        #imagen_11 = cv2.drawKeypoints(imagen_11, kp_11, imagen_11)

        kp_12, imagen_esquinas_12 = harris_kp(copia_imagen_12, threshold = 145)
        esquinas_12 = np.argwhere(imagen_esquinas_12 == 1)
        for esquina in esquinas_12:
            y, x = esquina
            cv2.circle(imagen_12, (x, y), radius=1, color=(255, 255, 255), thickness=-1)
        _, descriptores_12 = sift.compute(imagen_12, kp_12)
        #imagen_12 = cv2.drawKeypoints(imagen_12, kp_12, imagen_12)

        registrar(imagen_11, imagen_12, kp_11, kp_12, descriptores_11, descriptores_12)


    return

if __name__ == "__main__":
    ejercicio3()