import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt

def redimensionar(imagenes, indices_redimensionar):
    """
    Redimensiona las imágenes en el set que se encuentran en los índices especificados
    para que tengan el mismo tamaño que la primera imagen del set (imagen de referencia).
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        indices_redimensionar: Lista de índices que indican qué imágenes deben ser redimensionadas.
        
    Returns:
        Lista de imágenes donde las imágenes en los índices especificados han sido redimensionadas.
    """
    # Obtener las dimensiones de la primera imagen (de referencia)
    referencia_alto, referencia_ancho = imagenes[0].shape[:2]
    
    # Copiar el set original para no modificar el set original directamente
    imagenes_modificadas = imagenes.copy()
    
    # Redimensionar solo las imágenes en los índices proporcionados
    for i in indices_redimensionar:
        imagenes_modificadas[i] = cv2.resize(imagenes_modificadas[i], (referencia_ancho, referencia_alto))
    
    return imagenes_modificadas


def filtros(imagenes, sufijos):
    cantidad = len(imagenes)
    # Crear una figura ajustable en tamaño y número de subplots
    filtros_nombres = ["Promedio", "Mediana", "Gaussiano", "Bilateral", "NLM"]
    num_filtros = len(filtros_nombres)

    fig = plt.figure(figsize=(3 * cantidad, 3 * num_filtros))  # Ajuste del tamaño de la figura

    for i, imagen in enumerate(imagenes):
        # Aplicar cada filtro a la imagen
        filtradas = [
            cv2.blur(imagen, (3, 3)),
            cv2.medianBlur(imagen, 3),
            cv2.GaussianBlur(imagen, (5, 5), sigmaX=1),
            cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75),
            cv2.fastNlMeansDenoising(imagen, h=10, templateWindowSize=7, searchWindowSize=21)
        ]

        # Mostrar cada filtro en una nueva posición de subplot
        for j, (filtro_imagen, filtro_nombre) in enumerate(zip(filtradas, filtros_nombres)):
            ax_idx = j * cantidad + i + 1  # Índice único de subplot para cada imagen-filtro
            plt.subplot(num_filtros, cantidad, ax_idx)
            plt.imshow(cv2.cvtColor(filtro_imagen, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            # Añadir el nombre del filtro debajo de cada imagen
            plt.text(0.5, -0.05, filtro_nombre, ha='center', va='top', fontsize=15, transform=plt.gca().transAxes)

            # Añadir sufijo en la parte superior solo de la primera fila de cada columna
            if j == 0 and i < len(sufijos):
                plt.title(sufijos[i], fontsize=12, pad=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.001, wspace=0.1)  # Ajustar espaciado vertical (hspace) y horizontal (wspace)
    plt.show()



def soft_thresholding(coeffs, threshold):
    # Aplicar umbral suave a los coeficientes
    return pywt.threshold(coeffs, threshold, mode='soft')

def wavelet(imagenes, varianzas, sufijos=None):
    if sufijos is None:
        sufijos = [f"Imagen {i+1}" for i in range(len(imagenes))]

    for idx, (imagen, varianza) in enumerate(zip(imagenes, varianzas)):
        # Convertir la imagen a escala de grises si es a color
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen

        # Realizar la DWT hasta nivel 2
        coeffs = pywt.wavedec2(imagen_gris, 'haar', level=2)

        # Extraer coeficientes de nivel 2 y nivel 1
        LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs
        sigma = np.sqrt(varianza)
        # Calcular el umbral usando sigma
        threshold = sigma * np.sqrt(2 * np.log2(imagen_gris.size))
        print(f"Umbral calculado para {sufijos[idx]}: {threshold}")

        # Aplicar umbralización suave a los coeficientes de detalle de nivel 2
        LL2_thresh = soft_thresholding(LL2, threshold)
        LH2_thresh = soft_thresholding(LH2, threshold)
        HL2_thresh = soft_thresholding(HL2, threshold)
        HH2_thresh = soft_thresholding(HH2, threshold)

        # Reconstruir la imagen usando los coeficientes umbralizados
        coeffs_thresh = [LL2_thresh, (LH2_thresh, HL2_thresh, HH2_thresh), (LH1, HL1, HH1)]
        imagen_denoised = pywt.waverec2(coeffs_thresh, 'haar')
        imagen_denoised = np.clip(imagen_denoised, 0, 255)  # Limitar valores para visualización

        # Mostrar imagen ruidosa y reconstruida
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.title(f'{sufijos[idx]} - Ruidosa')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(imagen_denoised, cmap='gray')
        plt.title(f'{sufijos[idx]} - Reconstruida')
        plt.axis('off')
        
        plt.suptitle(f'Reducción de Ruido - {sufijos[idx]}')
        plt.tight_layout()
        plt.show()
