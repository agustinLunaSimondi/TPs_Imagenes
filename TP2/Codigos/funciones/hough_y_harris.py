import numpy as np
import cv2

def lineas_hough(imagen, lower_thersh = 50, upper_thresh = 150, longitud = 200):
    bordes = cv2.Canny(imagen, lower_thersh, upper_thresh)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, longitud, minLineLength=10, maxLineGap=10)
    return lineas


# k es un parámetro entre 0.04 y 0.06 para ajustar la sensibilidad
def harris(imagen, block_size = 8, ksize = 5, k = 0.04, threshold = False):
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
        threshold = 0.5* harris_corners.max()
    imagen_esquinas[harris_corners > threshold] = 1
    
    return imagen_esquinas