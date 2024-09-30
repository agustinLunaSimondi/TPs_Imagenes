import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detectar_puntos_interes(imagen, metodo='SIFT'):
    """
    Detecta puntos de interés en una imagen usando SIFT u ORB.
    
    Args:
        imagen: La imagen de entrada en formato BGR.
        metodo: El método a usar para detectar puntos de interés ('SIFT', 'ORB').
    
    Returns:
        keypoints: Lista de puntos clave detectados.
        output_image: Imagen con los puntos clave dibujados.
    """
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

    if metodo == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    elif metodo == 'ORB':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Dibujar los puntos clave en la imagen
    output_image = cv2.drawKeypoints(imagen, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    
    return keypoints, output_image

def comparar_metodos_puntos_interes(imagenes, sufijos, output_dir='./output/puntos_interes'):
    """
    Compara los métodos SIFT y ORB en las imágenes y guarda los resultados.
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        sufijos: Lista de sufijos para identificar cada imagen.
        output_dir: Directorio base para guardar los resultados de los puntos de interés.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe
    
    metodos = ['SIFT', 'ORB']  # Se elimina SURF

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
        else:
            # Crear directorio para la imagen
            imagen_dir = os.path.join(output_dir, sufijos[i])
            os.makedirs(imagen_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 5))
            
            for j, metodo in enumerate(metodos):
                try:
                    # Detectar puntos de interés con el método especificado
                    keypoints, output_image = detectar_puntos_interes(imagen, metodo=metodo)
                    
                    # Mostrar la imagen con los puntos de interés
                    plt.subplot(1, len(metodos), j + 1)
                    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                    plt.title(f"{metodo} - {sufijos[i]}")
                    plt.axis('off')
                    
                    # Guardar la imagen con los puntos de interés
                    output_path = os.path.join(imagen_dir, f"{sufijos[i]}_{metodo}.png")
                    cv2.imwrite(output_path, output_image)

                except cv2.error as e:
                    print(f"Error usando {metodo} en {sufijos[i]}: {e}")

            plt.tight_layout()
            plt.show()