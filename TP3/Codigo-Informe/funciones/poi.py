import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def detectar_puntos_interes(imagenes, metodo='SIFT'):
    """
    Detecta puntos de interés en un conjunto de imágenes usando SIFT u ORB.
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        metodo: El método a usar para detectar puntos de interés ('SIFT', 'ORB').
    
    Returns:
        keypoints_list: Lista de listas de puntos clave detectados para cada imagen.
        output_images: Lista de imágenes con los puntos clave dibujados.
    """
    keypoints_list = []
    output_images = []
    
    for imagen in imagenes:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        
        if metodo == 'SIFT':
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        elif metodo == 'ORB':
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Dibujar los puntos clave en la imagen
        output_image = cv2.drawKeypoints(imagen, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        
        # Almacenar resultados en las listas
        keypoints_list.append(keypoints)
        output_images.append(output_image)
    
    return keypoints_list, output_images


def obtener_caracteristicas(imagenes, sufijos, metodo='SIFT', output_dir='./TP3/output/puntos_interes'):
    """
    Compara los métodos SIFT y ORB en las imágenes y guarda los resultados.
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        sufijos: Lista de sufijos para identificar cada imagen.
        output_dir: Directorio base para guardar los resultados de los puntos de interés.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe
    
    keypoints_list, output_images = detectar_puntos_interes(imagenes, metodo=metodo)
    plt.figure(figsize=(10, 5))
    for i, output_image in enumerate(output_images):
        if output_image is None:
            print(f"Error: No se detectaron correctamente los puntos de la imagen {sufijos[i]}.tif")
        else:
            # Crear directorio para la imagen
            imagen_dir = os.path.join(output_dir, sufijos[i])
            os.makedirs(imagen_dir, exist_ok=True)
            
           
            
            try:
                # Mostrar la imagen con los puntos de interés
                plt.subplot(1, len(imagenes), i + 1)
                plt.title(f'{sufijos[i]}-{metodo}')
                plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))  # Acceder a cada imagen individualmente
                plt.axis('off')
                plt.tight_layout()
                
                # Guardar la imagen con los puntos de interés
                output_path = os.path.join(imagen_dir, f"{sufijos[i]}_{metodo}.png")
                cv2.imwrite(output_path, output_image)  # Acceder a cada imagen individualmente

            except cv2.error as e:
                print(f"Error usando {metodo} en {sufijos[i]}: {e}")

    
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def imagenes_registradas(imagenes, sufijos, referencia, matches_filtrados=None, cant_matches=50):
    sift = cv2.SIFT_create()
    # Detectar los puntos clave y calcular los descriptores para la imagen de referencia
    kp_f, descrip_f = sift.detectAndCompute(referencia, mask=None)

    for i, imagen in enumerate(imagenes):
        if i != 0:
            kp_m, descrip_m = sift.detectAndCompute(imagen, mask=None)

            if matches_filtrados is None:
                # Usamos un matcheador de fuerza bruta
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                # Encontrar los matches
                matches = bf.match(descrip_f, descrip_m)
                # Ordenar los matches por distancia
                matches = sorted(matches, key=lambda x: x.distance)
            else:
                matches = matches_filtrados

            # Filtrar matches válidos: asegurarse que los índices estén en rango
            matches_validos = [m for m in matches if m.trainIdx < len(kp_m) and m.queryIdx < len(kp_f)]

            if len(matches_validos) > 0:
                # Verificar que haya suficientes matches válidos para dibujar
                cant_matches_a_dibujar = min(len(matches_validos), cant_matches)

                if cant_matches_a_dibujar > 0:
                    # Dibujar las coincidencias usando los matches válidos
                    coincidencias = cv2.drawMatches(
                        imagen, kp_m, referencia, kp_f, matches_validos[:cant_matches_a_dibujar], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Obtener las coordenadas de los matches válidos
                    src_pts = np.float32([kp_f[m.queryIdx].pt for m in matches_validos[:cant_matches_a_dibujar]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_m[m.trainIdx].pt for m in matches_validos[:cant_matches_a_dibujar]]).reshape(-1, 1, 2)

                    # Calcular la homografía usando RANSAC
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

                    # Registrar la imagen usando la homografía
                    h, w = referencia.shape[:2]
                    imagen_registrada = cv2.warpPerspective(referencia, H, (w, h))

                    # Mostrar las coincidencias y la imagen registrada
                    plt.subplot(len(imagenes)-1, 2, 2*i-1)
                    plt.title(f'Coincidencias con {sufijos[i]}')
                    plt.imshow(coincidencias)
                    plt.axis('off')

                    plt.subplot(len(imagenes)-1, 2, 2*i)
                    plt.title(f'Registro con {sufijos[i]}')
                    plt.imshow(imagen_registrada, cmap='gray')
                    plt.axis('off')
                else:
                    print(f"No hay suficientes matches válidos para la imagen {sufijos[i]}")
            else:
                print(f"No se encontraron matches válidos para la imagen {sufijos[i]}")

    plt.tight_layout()
    plt.show()