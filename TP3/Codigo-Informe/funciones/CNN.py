import numpy as np
from scipy import fftpack
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform, img_as_float
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from scipy.optimize import differential_evolution
from scipy import fftpack
from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.generales import plot_1, plot_2
from funciones.MI import registrar_MI, fondo_blanco_a_negro
from funciones.metricas import calcular_mse_ssim


def cnn(lista_imagenes, umbral_ncc=0.6):
    # Cargar la imagen de referencia (la primera en la lista)
    img_ref = lista_imagenes[0]
    if img_ref is None:
        raise ValueError("La imagen de referencia no se pudo cargar.")
    
    # Asegurarse de que todas las imágenes sean del mismo tamaño que la referencia
    altura, ancho = img_ref.shape
    

    # Lista para almacenar las imágenes registradas
    imagenes_registradas = [img_ref.copy()]

    # Iterar sobre las demás imágenes en la lista
    for i in range(1, len(lista_imagenes)):
        img_movil = lista_imagenes[i]
        
        if img_movil is None:
            print(f"La imagen en la posición {i} no se pudo cargar.")
            continue

        # Calcular la correlación cruzada normalizada
        ncc = cv2.matchTemplate(img_movil, img_ref, method=cv2.TM_CCORR_NORMED)
        
        # Encontrar el valor máximo y su ubicación
        _, max_v, _, max_loc = cv2.minMaxLoc(ncc)
        
        # Verificar si el valor de NCC es suficientemente alto para considerar que hay coincidencia
        if max_v < umbral_ncc:
            print(f"Desplazamiento para la imagen {i} no confiable con NCC = {max_v}")
            continue

        # Coordenadas del punto óptimo de coincidencia (desplazamiento óptimo)
        topleft = max_loc
        print(f"Desplazamiento óptimo para la imagen {i}: {topleft} con valor de NCC = {max_v}")
        
        # Matriz de transformación para la traslación
        M = np.float32([[1, 0, topleft[0]], [0, 1, topleft[1]]])
        
        # Aplicar la transformación a la imagen móvil para alinear con la de referencia
        imagen_registrada = cv2.warpAffine(img_movil, M, (ancho, altura), flags=cv2.INTER_LINEAR)
        
        # Guardar la imagen registrada
        imagenes_registradas.append(imagen_registrada)
        
        # Mostrar resultados usando subplots
        plt.figure(figsize=(15, 5))
        
        # Mostrar la imagen de referencia
        plt.subplot(1, 3, 1)
        plt.imshow(img_ref, cmap='gray')
        plt.title("Imagen de Referencia")
        plt.axis('off')
        
        # Mostrar la imagen móvil (sin registrar)
        plt.subplot(1, 3, 2)
        plt.imshow(img_movil, cmap='gray')
        plt.title("Imagen Móvil (sin registrar)")
        plt.axis('off')
        
        # Mostrar la imagen móvil ya registrada
        plt.subplot(1, 3, 3)
        plt.imshow(imagen_registrada, cmap='gray')
        plt.title("Imagen Móvil (registrada)")
        plt.axis('off')
        
        plt.show()
    
    return imagenes_registradas
   


def cnn2(lista_imagenes):
    # Cargar la imagen de referencia (fija) desde la primera posición de la lista
    img_ref = lista_imagenes[0]
    
    
    # Normalizar y aplicar Gaussian Blur a la imagen de referencia
    img_ref = cv2.normalize(img_ref.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img_ref = cv2.GaussianBlur(img_ref, (5, 5), 0)
    
    # Dimensiones de la imagen de referencia
    alto, ancho = img_ref.shape
    
    # Iterar sobre las imágenes móviles y registrarlas
    for i in range(1, len(lista_imagenes)):
        img_mov = lista_imagenes[i]
        
        # Normalizar y aplicar Gaussian Blur a la imagen móvil
        img_mov = cv2.normalize(img_mov.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
        img_mov = cv2.GaussianBlur(img_mov, (5, 5), 0)
        
        # Búsqueda gruesa de rotación
        rotaciones = np.arange(-180, 180, 5)
        mejor_corr = -1
        mejor_ang = 0
        mejor_t = (0, 0)
        mejor_img_rotada = img_mov.copy()
        
        for ang in rotaciones:
            M_rot = cv2.getRotationMatrix2D((ancho/2, alto/2), ang, 1)
            img_rotada = cv2.warpAffine(img_mov, M_rot, (ancho, alto), flags=cv2.INTER_LINEAR)

            F_ref = fftpack.fft2(img_ref)
            F_rotada = fftpack.fft2(img_rotada)
            cross_power = F_ref * np.conj(F_rotada)
            cross_corr = fftpack.ifft2(cross_power)
            cross_corr = fftpack.fftshift(np.abs(cross_corr))
            
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            max_val = cross_corr[max_idx]

            if max_val > mejor_corr:
                mejor_corr = max_val
                mejor_ang = ang
                t_y = max_idx[0] - alto // 2
                t_x = max_idx[1] - ancho // 2
                mejor_t = (t_x, t_y)
                mejor_img_rotada = img_rotada.copy()

        # Búsqueda fina de rotación
        rotaciones_f = np.arange(mejor_ang - 5, mejor_ang + 6, 1)
        mejor_corr_f = -1
        mejor_ang_f = mejor_ang
        mejor_t_f = mejor_t
        mejor_img_rotada_f = mejor_img_rotada.copy()
        
        for ang in rotaciones_f:
            M_rot = cv2.getRotationMatrix2D((ancho/2, alto/2), ang, 1)
            img_rotada = cv2.warpAffine(img_mov, M_rot, (ancho, alto), flags=cv2.INTER_LINEAR)

            F_rotada = fftpack.fft2(img_rotada)
            cross_power = F_ref * np.conj(F_rotada)
            cross_corr = fftpack.ifft2(cross_power)
            cross_corr = fftpack.fftshift(np.abs(cross_corr))
            
            max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
            max_val = cross_corr[max_idx]

            if max_val > mejor_corr_f:
                mejor_corr_f = max_val
                mejor_ang_f = ang
                t_y = max_idx[0] - alto // 2
                t_x = max_idx[1] - ancho // 2
                mejor_t_f = (t_x, t_y)
                mejor_img_rotada_f = img_rotada.copy()

        # Aplicar la rotación y traslación óptimas
        M_rot_final = cv2.getRotationMatrix2D((ancho/2, alto/2), mejor_ang_f, 1)
        M_rot_final[0, 2] += mejor_t_f[0]
        M_rot_final[1, 2] += mejor_t_f[1]
        img_registrada = cv2.warpAffine(img_mov, M_rot_final, (ancho, alto), flags=cv2.INTER_LINEAR)

        # Calcular métricas de calidad (MSE y SSIM)
        
        mse = np.mean((img_ref - img_registrada) ** 2)
        ssim_index, _ = ssim(img_ref, img_registrada, full=True, data_range=1.0)

        # Visualizar el registro
        plt.figure(figsize=(18, 6))
        
        # Imagen de referencia
        plt.subplot(1, 4, 1)
        plt.imshow(img_ref, cmap='gray')
        plt.title('Imagen de Referencia')
        plt.axis('off')

        # Imagen de referencia
        plt.subplot(1, 4, 2)
        plt.imshow(img_mov, cmap='gray')
        plt.title('Imagen movil')
        plt.axis('off')

        # Imagen registrada
        plt.subplot(1, 4, 3)
        plt.imshow(img_registrada, cmap='gray')
        plt.title(f'Imagen Móvil Registrada\nRotación: {mejor_ang_f}°\nTraslación: {mejor_t_f}')
        plt.axis('off')

        # Diferencia entre imágenes
        diferencia = cv2.absdiff((img_ref * 255).astype(np.uint8), (img_registrada * 255).astype(np.uint8))
        plt.subplot(1, 4, 4)
        plt.imshow(diferencia, cmap='gray')
        plt.title('Diferencia entre Imágenes')
        plt.axis('off')
        plt.show()

        # Mostrar métricas de calidad
        print(f"Imagen {i} - Error Cuadrático Medio (MSE): {mse:.4f}")
        print(f"Imagen {i} - Índice de Similitud Estructural (SSIM): {ssim_index:.4f}")
