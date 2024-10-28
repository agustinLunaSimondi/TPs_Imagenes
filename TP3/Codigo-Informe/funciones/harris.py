import numpy as np
import cv2
import matplotlib.pyplot as plt
from funciones.metricas import calcular_mse_ssim

def harris_kp(img, block_size = 8, ksize = 5, k = 0.04, threshold = False):
    imagen = img.copy()
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    imagen_float = np.float32(imagen_gris)
    harris_corners = cv2.cornerHarris(imagen_float, block_size, ksize, k)
    harris_corners = cv2.dilate(harris_corners, None)
    harris_corners = cv2.normalize(harris_corners, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    filas = imagen_gris.shape[0]
    columnas = imagen_gris.shape[1]
    imagen_esquinas = np.zeros([filas, columnas])
    if not threshold:
        threshold = 0.1* harris_corners.max()

    imagen_esquinas[harris_corners > threshold] = 1
    keypoints = np.argwhere(harris_corners > threshold)
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]

    return keypoints, imagen_esquinas


def registrar(img_f, img_m, kp_f, kp_m, descrip_f, descrip_m, graficar = True, calcular_metricas = True):
    imagen_f = img_f.copy()
    imagen_m = img_m.copy()

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descrip_m, descrip_f)
    matches_filtrados = sorted(matches, key=lambda x: x.distance)
    #matches_filtrados = [m for m in matches if m.distance < 0.7 * min([match.distance for match in matches])]

    if len(matches_filtrados) > 4:
        pa = np.float32([kp_f[m.trainIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)
        pb = np.float32([kp_m[m.queryIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)

        homografia, mask = cv2.findHomography(pb, pa, cv2.RANSAC, 5.0)

        altura = imagen_f.shape[0]
        ancho = imagen_f.shape[1]
        imagen_registrada = cv2.warpPerspective(imagen_m, homografia, (ancho, altura), flags=cv2.INTER_LINEAR)
        
    coincidencias = cv2.drawMatches(imagen_m, kp_m,  imagen_f, kp_f, matches_filtrados, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if graficar:
        plt.figure(figsize=(20,10))

        plt.subplot(1, 2, 1)
        plt.title('Imagen Fija')
        plt.imshow(imagen_f, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Imagen movil')
        plt.imshow(imagen_m, cmap='gray')
        plt.axis('off')

        plt.show()

        plt.figure(figsize=(20,10))

        # Mostrar las coincidencias
        plt.subplot(1, 2, 1)
        plt.title('Coincidencias SIFT con FLANN y Ratio de Lowe')
        plt.imshow(coincidencias, cmap='gray')
        plt.axis('off')

        # Mostrar la imagen registrada
        plt.subplot(1, 2, 2)
        plt.title('Imagen Registrada')
        plt.imshow(imagen_registrada, cmap='gray')
        plt.axis('off')

        plt.show()

    if calcular_metricas:
        calcular_mse_ssim(imagen_f, imagen_registrada)

    return