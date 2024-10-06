import numpy as np
import matplotlib.pyplot as plt
import cv2

def obtener_contornos(imagen, umbral_bin):
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    _, imagen_binaria = cv2.threshold(imagen_gris, umbral_bin, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def visualizar_contornos(imagen, contornos, idx_contorno = -1): # idx negativo los dibuja todos
    imagen_contorno = imagen.copy()
    cv2.drawContours(imagen_contorno, contornos, idx_contorno, (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(imagen_contorno, cv2.COLOR_BGR2RGB))
    plt.title('Contornos seleccionados')
    plt.show()
    return imagen_contorno

def segmentar_contorno(imagen, contorno):
    mask = np.zeros_like(imagen)
    cv2.drawContours(mask, [contorno], -1, (255, 255, 255), thickness=cv2.FILLED)
    imagen_segmentada = cv2.bitwise_and(imagen, mask)
    return imagen_segmentada