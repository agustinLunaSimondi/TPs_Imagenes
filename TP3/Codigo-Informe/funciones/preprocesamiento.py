import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from numpy.fft import fftshift, fft2, ifft2
from scipy.ndimage import laplace
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import img_as_float

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



def wavelet2(img_with_noise_np):
    # Asegurarse de que la imagen esté en formato de punto flotante y normalizada entre [0, 1]
    img_with_noise_np = img_as_float(img_with_noise_np)
    
    # Calcular la estimación de sigma del ruido
    sigma_est = estimate_sigma(img_with_noise_np)
    
    # Definir el eje de canal basado en la dimensionalidad de la imagen
    channel_axis = -1 if img_with_noise_np.ndim == 3 else None
    
    # Aplicar el filtro de ondículas con channel_axis en lugar de multichannel
    wavelet_filtered = denoise_wavelet(
        img_with_noise_np, 
        sigma=sigma_est, 
        mode='soft', 
        wavelet_levels=3, 
        channel_axis=channel_axis,  # Usamos channel_axis en vez de multichannel
        rescale_sigma=True
    )
    

    plt.figure(figsize=(8, 4))
        
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_noise_np, cmap='gray')
    plt.title(' - Ruidosa')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(wavelet_filtered, cmap='gray')
    plt.title('- Reconstruida')
    plt.axis('off')
   
    plt.tight_layout()
    plt.show()


    return wavelet_filtered



def soft_thresholding(coeffs, threshold):
    # Aplicar umbral suave a los coeficientes
    return pywt.threshold(coeffs, threshold, mode='soft')

def wavelet(imagenes, varianzas,indices_modificar =None ,sufijos=None):
    if sufijos is None:
        sufijos = [f"Imagen {i+1}" for i in range(len(imagenes))]
    image_wavelet = []
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
        image_wavelet.append(imagen_denoised)

    imagenes_retornadas=[]
    
    if(indices_modificar == None):
        print("No hay imagenes a devolve")
    else:
        for i in indices_modificar:
            image_wavelet[i] = cv2.normalize(image_wavelet[i], None, 0, 255, cv2.NORM_MINMAX)
            image_wavelet[i] = image_wavelet[i].astype('uint8')
            imagenes_retornadas.append(image_wavelet[i])
        
        return imagenes_retornadas




def nlm(imagenes, indices_modificar):
    """
    Redimensiona las imágenes en el set que se encuentran en los índices especificados
    para que tengan el mismo tamaño que la primera imagen del set (imagen de referencia).
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        indices_redimensionar: Lista de índices que indican qué imágenes deben ser redimensionadas.
        
    Returns:
        Lista de imágenes donde las imágenes en los índices especificados han sido modificadas.
    """
  
    
    # Copiar el set original para no modificar el set original directamente
    imagenes_modificadas = imagenes.copy()
    
    # Redimensionar solo las imágenes en los índices proporcionados
    for i in indices_modificar:
        imagenes_modificadas[i] = cv2.fastNlMeansDenoising(imagenes_modificadas[i], h=10, templateWindowSize=7, searchWindowSize=21)
    
    return imagenes_modificadas



    
def filtrado_fourier(imagenes):
    for img in imagenes:
        # 1. Calcular la Transformada de Fourier de la imagen y desplazar el cero de frecuencia al centro
        f_transform = fftshift(fft2(img))
        magnitude_spectrum = np.log(1 + np.abs(f_transform))

        # 2. Crear un filtro de paso bajo circular
        rows, cols = img.shape
        crow, ccol = rows // 2 , cols // 2
        mask = np.zeros((rows, cols), dtype=np.float32)
        r = 30  # Radio del filtro de paso bajo
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r * r
        mask[mask_area] = 1

        # 3. Aplicar el filtro de paso bajo
        filtered_transform = f_transform * mask

        # 4. Transformada Inversa para obtener la imagen filtrada
        filtered_img = np.abs(ifft2(fftshift(filtered_transform)))

        # Mostrar el espectro de frecuencia original, el filtro aplicado y la imagen resultante
        plt.figure(figsize=(15, 5))

        # Espectro de frecuencia original
        plt.subplot(1, 3, 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("Espectro de Frecuencia Original")
        plt.axis('off')

        # Filtro de paso bajo aplicado en el espectro
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Filtro de Paso Bajo")
        plt.axis('off')

        # Imagen después del filtrado en Fourier
        plt.subplot(1, 3, 3)
        plt.imshow(filtered_img, cmap='gray')
        plt.title("Imagen Filtrada (Fourier)")
        plt.axis('off')

        plt.show()

def filtrado_fourier2(img):

   # 1. Calcular la Transformada de Fourier de la imagen y desplazar el cero de frecuencia al centro
    f_transform = fftshift(fft2(img))
    magnitude_spectrum = np.log(1 + np.abs(f_transform))

    # 2. Crear una máscara notch para eliminar la línea diagonal en el espectro de frecuencias
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    # Crear la máscara inicial (una matriz de unos)
    notch_mask = np.ones((rows, cols), dtype=np.float32)

    # Configuración para el filtro notch
    num_notches = 125   # Número de puntos a bloquear en la línea diagonal
    line_slope = 0.3   # Pendiente aproximada de la línea diagonal en el espectro
    line_offset = 3   # Tamaño del área notch alrededor de cada punto
    protect_radius = 20  # Radio de protección central

    # Bloquear una banda diagonal en el espectro fuera del rango central
    for i in range(-num_notches, num_notches + 1):
        y = int(crow + i * line_slope)
        x = int(ccol + i)
        
        # Saltar el área dentro del radio de protección central
        if (x - ccol) ** 2 + (y - crow) ** 2 > protect_radius ** 2:
            # Bloquear en ambas direcciones de la línea diagonal
            notch_mask[y-line_offset:y+line_offset, x-line_offset:x+line_offset] = 0
            notch_mask[rows-y-line_offset:rows-y+line_offset, cols-x-line_offset:cols-x+line_offset] = 0

    # 3. Aplicar la máscara notch en el espectro de Fourier
    filtered_transform_notch = f_transform * notch_mask

    # 4. Transformada inversa para obtener la imagen filtrada
    filtered_img_notch = np.abs(ifft2(fftshift(filtered_transform_notch)))

    # 5. Mostrar el espectro modificado y la imagen resultante
    plt.figure(figsize=(15, 5))

    # Espectro de frecuencia original
    plt.subplot(1, 3, 1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Espectro de Frecuencia Original")
    plt.axis('off')

    # Máscara notch aplicada en el espectro
    plt.subplot(1, 3, 2)
    plt.imshow(notch_mask, cmap='gray')
    plt.title("Filtro Notch con Protección Central")
    plt.axis('off')

    # Imagen después del filtrado notch en Fourier
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_img_notch, cmap='gray')
    plt.title("Imagen Filtrada (Notch con Protección Central)")
    plt.axis('off')

    plt.show()

def laplaciano(imagen):
    
    

    image_blurry = imagen

    # Aplicar el filtro Laplaciano
    laplacian_filtered = laplace(image_blurry)

    # Realzar detalles sumando el filtro laplaciano a la imagen original
    sharpened_image = image_blurry + 0.5 * laplacian_filtered  # Ajuste de intensidad

    # Visualizar los resultados
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(image_blurry, cmap='gray')
    ax[0].set_title("Imagen Original (Desenfocada)")
    ax[0].axis('off')

    ax[1].imshow(laplacian_filtered, cmap='gray')
    ax[1].set_title("Filtro Laplaciano")
    ax[1].axis('off')

    ax[2].imshow(sharpened_image, cmap='gray')
    ax[2].set_title("Imagen Mejorada (Filtro Laplaciano)")
    ax[2].axis('off')

    plt.show()
    return sharpened_image

def aplicar_clahe(imagen, clipLimit=2.0, tileGridSize=(8, 8)):
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Convertir a uint8 si es necesario
    imagen = imagen.astype('uint16')
    
    # Aplicar CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    imagen_clahe = clahe.apply(imagen)
    
    # Visualización
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(imagen, cmap='gray')
    ax[1].imshow(imagen_clahe, cmap='gray')
    plt.show()

    imagen_clahe = cv2.normalize(imagen_clahe, None, 0, 255, cv2.NORM_MINMAX)
    imagen_clahe = imagen_clahe.astype('uint8')
    return imagen_clahe
    


def filtros_clahe(imagenes, sufijos, clipLimit=2.0, tileGridSize=(8, 8)):
    cantidad = len(imagenes)
    # Definir nombres y filtros
    filtros_nombres = ["Promedio", "Mediana", "Gaussiano", "Bilateral", "NLM"]
    num_filtros = len(filtros_nombres)

    fig = plt.figure(figsize=(6 * cantidad, 3 * num_filtros))  # Tamaño ajustado para mostrar dos columnas por filtro

    for i, imagen in enumerate(imagenes):
        # Aplicar cada filtro a la imagen
        filtradas = [
            cv2.blur(imagen, (3, 3)),
            cv2.medianBlur(imagen, 3),
            cv2.GaussianBlur(imagen, (5, 5), sigmaX=1),
            cv2.bilateralFilter(imagen, d=9, sigmaColor=75, sigmaSpace=75),
            cv2.fastNlMeansDenoising(imagen, h=10, templateWindowSize=7, searchWindowSize=21)
        ]

        # Aplicar CLAHE a cada imagen filtrada
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        filtradas_clahe = [clahe.apply(filtro) for filtro in filtradas]

        # Mostrar cada filtro y su versión con CLAHE
        for j, (filtro_imagen, filtro_imagen_clahe, filtro_nombre) in enumerate(zip(filtradas, filtradas_clahe, filtros_nombres)):
            # Mostrar la imagen solo con el filtro
            ax_idx_filtro = j * cantidad * 2 + i * 2 + 1
            plt.subplot(num_filtros, cantidad * 2, ax_idx_filtro)
            plt.imshow(cv2.cvtColor(filtro_imagen, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.text(0.5, -0.1, filtro_nombre, ha='center', va='top', fontsize=12, transform=plt.gca().transAxes)

            # Mostrar la imagen con filtro + CLAHE
            ax_idx_clahe = ax_idx_filtro + 1
            plt.subplot(num_filtros, cantidad * 2, ax_idx_clahe)
            plt.imshow(cv2.cvtColor(filtro_imagen_clahe, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.text(0.5, -0.1, filtro_nombre + "+CLAHE", ha='center', va='top', fontsize=12, transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.1)
    plt.show()