import cv2
import matplotlib.pyplot as plt
import numpy as np


def cnn(lista_imagenes,sufijos):
    # Cargar la imagen de referencia (la primera en la lista)
    referencia = lista_imagenes[0]
    if referencia is None:
        raise ValueError("La imagen de referencia no se pudo cargar.")
    
    # Dimensiones de la imagen de referencia
    altura, ancho = referencia.shape
    
    # Lista para almacenar las imágenes registradas
    imagenes_registradas = [referencia]

    # Iterar sobre las demás imágenes en la lista
    for i in range(1, len(lista_imagenes)):
        movil = lista_imagenes[i]
        
        if movil is None:
            print(f"La imagen en la posición {i} no se pudo cargar.")
            continue

        # Calcular la correlación cruzada normalizada
        ncc = cv2.matchTemplate(movil, referencia, method=cv2.TM_CCORR_NORMED)
        
        # Encontrar el valor máximo y su ubicación
        _, max_v, _, max_loc = cv2.minMaxLoc(ncc)
        
        # Coordenadas del punto óptimo de coincidencia (desplazamiento óptimo)
        topleft = max_loc
        print(f"Desplazamiento óptimo para la imagen {i}: {topleft} con valor de NCC = {max_v}")
        
        # Matriz de transformación para la traslación
        M = np.float32([[1, 0, topleft[0]], [0, 1, topleft[1]]])
        
        # Aplicar la transformación a la imagen móvil para alinear con la de referencia
        imagen_registrada = cv2.warpAffine(movil, M, (ancho, altura), flags=cv2.INTER_LINEAR)
        
        # Guardar la imagen registrada
        imagenes_registradas.append(imagen_registrada)

        plt.subplot(1, 3, 1)
        plt.imshow(referencia, cmap='gray')
        plt.title(f'Referencia')
        plt.axis('off')


        plt.subplot(1, 3, 2)
        plt.imshow(movil, cmap='gray')
        plt.title(f'{sufijos[i]} - movil')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(imagen_registrada, cmap='gray')
        plt.title(f'{sufijos[i]} - Registrada')
        plt.axis('off')
    
    return imagenes_registradas