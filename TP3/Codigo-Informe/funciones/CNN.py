import numpy as np
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




def registrar_cnn(imagen1, imagen2, graficar = True, calcular_metricas = True):
    img_ref = imagen1.copy()
    img_mov = imagen2.copy()

    img_ref = cv2.normalize(img_ref.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img_mov = cv2.normalize(img_mov.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # 5. Definir rango de rotaciones para búsqueda Coarse
    rotaciones = np.arange(-30, 31, 5)  # De -30° a 30° en pasos de 5°
    mejor_corr = -1
    mejor_ang = 0
    mejor_t = (0, 0)
    mejor_img_rotada = img_mov.copy()

    # Obtener dimensiones de la imagen
    alto, ancho = img_ref.shape

    # 6. Búsqueda: Rotaciones amplias con pasos grandes
    for ang in rotaciones:
        # Obtener la matriz de rotación
        M_rot = cv2.getRotationMatrix2D((ancho/2, alto/2), ang, 1)  # Centro, ángulo, escala=1

        # Rotar la imagen a registrar
        img_rotada = cv2.warpAffine(img_mov, M_rot, (ancho, alto), flags=cv2.INTER_LINEAR)

        # Calcular la correlación cruzada usando FFT
        # FFT de ambas imágenes
        F_ref = fftpack.fft2(img_ref)
        F_rotada = fftpack.fft2(img_rotada)

        # Producto cruzado en el dominio de la frecuencia
        cross_power = F_ref * np.conj(F_rotada)

        # Inversa de la FFT para obtener la correlación en el dominio espacial
        cross_corr = fftpack.ifft2(cross_power)

        # Centrar la correlación
        cross_corr = fftpack.fftshift(cross_corr)
        cross_corr = np.abs(cross_corr)

        # Encontrar el pico de correlación
        max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        max_val = cross_corr[max_idx]

        # Actualizar si se encuentra una mejor correlación
        if max_val > mejor_corr:
            mejor_corr = max_val
            mejor_ang = ang
            # Calcular desplazamiento relativo al centro
            t_y = max_idx[0] - alto//2
            t_x = max_idx[1] - ancho//2
            mejor_t = (t_x, t_y)
            mejor_img_rotada = img_rotada.copy()

    print(f"Etapa 1 - Mejor Correlación: {mejor_corr:.4f}, Ángulo: {mejor_ang}°, Shift: {mejor_t}")

    # 7. Búsqueda Fina: Rotaciones refinadas alrededor del mejor ángulo encontrado
    rotaciones_f = np.arange(mejor_ang - 5, mejor_ang + 6, 1)  # ±5° en pasos de 1°
    mejor_corr_f = -1
    mejor_ang_f = mejor_ang
    mejor_t_f = mejor_t
    mejor_img_rotada_f = mejor_img_rotada.copy()

    for ang in rotaciones_f:
        # Obtener la matriz de rotación
        M_rot = cv2.getRotationMatrix2D((ancho/2, alto/2), ang, 1)

        # Rotar la imagen a registrar
        img_rotada = cv2.warpAffine(img_mov, M_rot, (ancho, alto), flags=cv2.INTER_LINEAR)

        # Calcular la correlación cruzada usando FFT
        F_ref = fftpack.fft2(img_ref)
        F_rotada = fftpack.fft2(img_rotada)

        cross_power = F_ref * np.conj(F_rotada)
        cross_corr = fftpack.ifft2(cross_power)
        cross_corr = fftpack.fftshift(cross_corr)
        cross_corr = np.abs(cross_corr)

        max_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        max_val = cross_corr[max_idx]

        if max_val > mejor_corr_f:
            mejor_corr_f = max_val
            mejor_ang_f = ang
            t_y = max_idx[0] - alto//2
            t_x = max_idx[1] - ancho//2
            mejor_t_f = (t_x, t_y)
            mejor_img_rotada_f = img_rotada.copy()

    print(f"Etapa 2 - Mejor Correlación: {mejor_corr_f:.4f}, Ángulo: {mejor_ang_f}°, Shift: {mejor_t_f}")

    # 8. Aplicar la rotación y traslación óptimas a la imagen a registrar
    # Crear la matriz de transformación afín con rotación y traslación
    # Primero, rotación
    M_rot_final = cv2.getRotationMatrix2D((ancho/2, alto/2), mejor_ang_f, 1)

    # Añadir traslación a la matriz de rotación
    M_rot_final[0, 2] += mejor_t_f[0]
    M_rot_final[1, 2] += mejor_t_f[1]

    # Aplicar la transformación
    img_registrada = cv2.warpAffine(img_mov, M_rot_final, (ancho,alto), flags=cv2.INTER_LINEAR)

    # 9. Visualizar el resultado final de la registración
    if graficar:
        plt.figure(figsize=(18, 6))

        # Imagen de referencia
        plt.subplot(1, 3, 1)
        plt.imshow(img_ref, cmap='gray')
        plt.title('Imagen de Referencia')
        plt.axis('off')

        # Imagen registrada (transformada y trasladada)
        plt.subplot(1, 3, 2)
        plt.imshow(img_registrada, cmap='gray')
        plt.title(f'Imagen Registrada\nRotación: {mejor_ang_f}°\nTraslación: {mejor_t_f}')
        plt.axis('off')

        # Diferencia entre imágenes
        diferencia = cv2.absdiff(img_ref, img_registrada)
        plt.subplot(1, 3, 3)
        plt.imshow(diferencia, cmap='gray')
        plt.title('Diferencia entre Imágenes')
        plt.axis('off')

        plt.show()

    # 10. Calcular y mostrar métricas de calidad
    if calcular_metricas:
        calcular_mse_ssim(img_ref, img_registrada)

    return