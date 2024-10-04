import numpy as np
import matplotlib.pyplot as plt
import cv2

def obtener_contornos(imagen, umbral_bin):
    _, imagen_binaria = cv2.threshold(imagen, umbral_bin, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contornos


def visualizar_contornos(imagen, contornos, idx_contorno = -1): # idx negativo los dibuja todos
    imagen_contorno = imagen.copy()
    cv2.drawContours(imagen_contorno, contornos, idx_contorno, (0, 255, 0), 2) # dibuja en verde
    plt.imshow(cv2.cvtColor(imagen_contorno, cv2.COLOR_BGR2RGB))
    plt.title('Contornos seleccionados')
    plt.show()
    return imagen_contorno