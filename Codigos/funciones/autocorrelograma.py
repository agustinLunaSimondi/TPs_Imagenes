import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from funciones.guardar import guardar_imagen

def autocorrelograma_por_canal(imagenes, sufijos, distancias=[1, 3, 5], n_bins=64, output_dir='./output/autocorrelogramas'):
    """
    Calcula y guarda los autocorrelogramas para cada canal de color en el espacio RGB.

    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        sufijos: Lista de sufijos para identificar cada imagen.
        distancias: Lista de distancias a considerar para el autocorrelograma.
        n_bins: Número de niveles de cuantización por canal (default 64).
        output_dir: Directorio base para guardar los autocorrelogramas.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
        else:
            # Crear directorio para la imagen
            imagen_dir = os.path.join(output_dir, sufijos[i])
            os.makedirs(imagen_dir, exist_ok=True)

            # Convertir la imagen de BGR a RGB
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

            # Dividir la imagen en los tres canales R, G, B
            canales = ['R', 'G', 'B']
            autocorrelogramas = {canal: np.zeros((n_bins, len(distancias))) for canal in canales}

            # Cuantizar la imagen en niveles por canal
            factor = 256 // n_bins
            imagen_cuantizada = (imagen_rgb // factor).astype(np.int32)

            # Para cada canal (R, G, B)
            for i, canal in enumerate(canales):
                canal_img = imagen_cuantizada[:, :, i]  # Extraer el canal actual

                # Para cada distancia, calcular el autocorrelograma
                for k, d in enumerate(distancias):
                    autocorrelograma = np.zeros(n_bins)

                    # Recorrer todos los píxeles de la imagen
                    for x in range(canal_img.shape[0]):
                        for y in range(canal_img.shape[1]):
                            color = canal_img[x, y]  # Color del píxel actual

                            # Recorrer los vecinos dentro de la distancia d
                            for dx in range(-d, d+1):
                                for dy in range(-d, d+1):
                                    if dx == 0 and dy == 0:
                                        continue
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < canal_img.shape[0] and 0 <= ny < canal_img.shape[1]:
                                        # Contar si el vecino tiene el mismo color
                                        if canal_img[nx, ny] == color:
                                            autocorrelograma[color] += 1

                    # Guardar el autocorrelograma para esta distancia
                    autocorrelogramas[canal][:, k] = autocorrelograma

            # Mostrar y guardar los autocorrelogramas por canal y distancia
            plt.figure(figsize=(15, 5))
            for i, canal in enumerate(canales):
                for k, d in enumerate(distancias):
                    plt.subplot(3, len(distancias), i * len(distancias) + k + 1)
                    plt.bar(range(n_bins), autocorrelogramas[canal][:, k])
                    plt.title(f"{canal} Canal - Distancia {d}")
                    plt.xlabel("Nivel de Color")
                    plt.ylabel("Conteo")
            
            plt.tight_layout()

            # Guardar la figura en el directorio correspondiente
            autocorrelograma_path = os.path.join(imagen_dir, f"{sufijos[i]}_autocorrelograma.png")
            plt.savefig(autocorrelograma_path)
            plt.show()

    return autocorrelogramas