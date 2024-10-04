import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from guardar import guardar_imagen
except ImportError:
    from funciones.guardar import guardar_imagen

def calcular_distancias(imagen):
    """
    Calcula las distancias basadas en el tamaño de la imagen.
    
    Args:
        imagen: La imagen de entrada en formato BGR.
    
    Returns:
        distancias: Lista de distancias calculadas.
    """
    alto, ancho = imagen.shape[:2]
    menor_lado = min(alto, ancho)
    return [1, menor_lado // 4, menor_lado // 2]

def autocorrelograma_por_canal(imagenes, sufijos, n_bins=64, output_dir='./output/autocorrelogramas'):
    """
    Calcula y guarda los autocorrelogramas para cada canal de color en el espacio RGB.

    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        sufijos: Lista de sufijos para identificar cada imagen.
        n_bins: Número de niveles de cuantización por canal (default 64).
        output_dir: Directorio base para guardar los autocorrelogramas.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
            continue

        # Crear directorio para la imagen
        imagen_dir = os.path.join(output_dir, sufijos[i])
        os.makedirs(imagen_dir, exist_ok=True)

        # Convertir la imagen de BGR a RGB
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Calcular las distancias en base al tamaño de la imagen
        distancias = calcular_distancias(imagen)

        # Dividir la imagen en los tres canales R, G, B
        canales = ['R', 'G', 'B']
        autocorrelogramas = {canal: np.zeros((n_bins, len(distancias))) for canal in canales}

        # Cuantizar la imagen en niveles por canal
        factor = 256 // n_bins
        imagen_cuantizada = (imagen_rgb // factor).astype(np.int32)

        # Para cada canal (R, G, B)
        for c, canal in enumerate(canales):
            canal_img = imagen_cuantizada[:, :, c]  # Extraer el canal actual

            # Para cada distancia, calcular el autocorrelograma
            for k, d in enumerate(distancias):
                autocorrelograma = np.zeros(n_bins)

                # Recorrer todos los píxeles de la imagen
                for x in range(canal_img.shape[0]):
                    for y in range(canal_img.shape[1]):
                        color = canal_img[x, y]  # Color del píxel actual

                        # Recorrer los vecinos dentro de la distancia d
                        for dx in range(-d, d + 1):
                            for dy in range(-d, d + 1):
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
        for c, canal in enumerate(canales):
            for k, d in enumerate(distancias):
                plt.subplot(3, len(distancias), c * len(distancias) + k + 1)
                plt.bar(range(n_bins), autocorrelogramas[canal][:, k])
                plt.title(f"{canal} Canal - Distancia {d} - {sufijos[i]}.tif")
                plt.xlabel("Nivel de Color")
                plt.ylabel("Conteo")
        
        plt.tight_layout()

        # Guardar la figura en el directorio correspondiente
        autocorrelograma_path = os.path.join(imagen_dir, f"{sufijos[i]}_autocorrelograma.png")
        plt.savefig(autocorrelograma_path)
        plt.show()

    return autocorrelogramas

def autocorrelograma_rgb(imagenes, sufijos, n_bins=64, output_dir='./output/autocorrelogramas'):
    """
    Calcula y guarda el autocorrelograma para el conjunto RGB en una imagen.

    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        sufijos: Lista de sufijos para identificar cada imagen.
        n_bins: Número de niveles de cuantización por canal (default 64).
        output_dir: Directorio base para guardar los autocorrelogramas.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
            continue

        # Crear directorio para la imagen
        imagen_dir = os.path.join(output_dir, sufijos[i])
        os.makedirs(imagen_dir, exist_ok=True)

        # Convertir la imagen de BGR a RGB
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # Calcular las distancias en base al tamaño de la imagen
        distancias = calcular_distancias(imagen)

        # Cuantizar la imagen en niveles RGB
        factor = 256 // n_bins
        imagen_cuantizada = (imagen_rgb // factor).astype(np.int32)

        # Crear el autocorrelograma para cada distancia
        autocorrelograma_total = np.zeros((n_bins, len(distancias)))

        # Para cada distancia
        for k, d in enumerate(distancias):
            # Recorrer todos los píxeles de la imagen
            for x in range(imagen_cuantizada.shape[0]):
                for y in range(imagen_cuantizada.shape[1]):
                    color = imagen_cuantizada[x, y]  # Color del píxel actual (R, G, B)

                    # Recorrer los vecinos dentro de la distancia d
                    for dx in range(-d, d + 1):
                        for dy in range(-d, d + 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < imagen_cuantizada.shape[0] and 0 <= ny < imagen_cuantizada.shape[1]:
                                vecino_color = imagen_cuantizada[nx, ny]  # Color del vecino
                                if np.array_equal(vecino_color, color):  # Solo sumar si el color del vecino es igual al actual
                                    autocorrelograma_total[color, k] += 1

        # Mostrar y guardar los autocorrelogramas para las diferentes distancias
        plt.figure(figsize=(15, 5))
        for k, d in enumerate(distancias):
            plt.subplot(1, len(distancias), k + 1)
            plt.bar(range(n_bins), autocorrelograma_total[:, k])
            plt.title(f"Distancia {d} - {sufijos[i]}.tif")
            plt.xlabel("Nivel de Color")
            plt.ylabel("Conteo")

        plt.tight_layout()

        # Guardar la figura en el directorio correspondiente
        autocorrelograma_path = os.path.join(imagen_dir, f"{sufijos[i]}_autocorrelograma_rgb.png")
        plt.savefig(autocorrelograma_path)
        plt.show()

    return autocorrelograma_total