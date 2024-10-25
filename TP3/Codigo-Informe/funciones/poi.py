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



def imagenes_registradas2(imagenes, sufijos, referencia, matches_filtrados=None, cant_matches=50):
    sift = cv2.SIFT_create()
    
    # Detectar los puntos clave y calcular los descriptores para la imagen de referencia
    kp_f, descrip_f = sift.detectAndCompute(referencia, mask=None)
    print(f"Puntos clave en la imagen de referencia: {len(kp_f)}")

    for i, imagen in enumerate(imagenes):
        if i != 0:
            kp_m, descrip_m = sift.detectAndCompute(imagen, mask=None)
            print(f"Puntos clave en la imagen {sufijos[i]}: {len(kp_m)}")

            if matches_filtrados is None:
                # Usamos un matcheador de fuerza bruta
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                # Encontrar los matches
                matches = bf.match(descrip_f, descrip_m)
                # Ordenar los matches por distancia
                matches = sorted(matches, key=lambda x: x.distance)
            else:
                matches = matches_filtrados

            # Filtrar matches válidos asegurando que los índices estén dentro de los límites
            matches_validos = [
                m for m in matches
                if 0 <= m.trainIdx < len(kp_m) and 0 <= m.queryIdx < len(kp_f)
            ]
            print(f"Matches válidos en la imagen {sufijos[i]}: {len(matches_validos)}")

            if len(matches_validos) > 0:
                # Verificar que haya suficientes matches válidos para dibujar
                cant_matches_a_dibujar = min(len(matches_validos), cant_matches)

                if cant_matches_a_dibujar > 0:
                    try:
                        # Revisar los matches antes de dibujar
                        checked_matches = []
                        for m in matches_validos[:cant_matches_a_dibujar]:
                            if 0 <= m.trainIdx < len(kp_m) and 0 <= m.queryIdx < len(kp_f):
                                checked_matches.append(m)

                        # Dibujar las coincidencias usando los matches válidos verificados
                        coincidencias = cv2.drawMatches(
                            imagen, kp_m, referencia, kp_f, checked_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )

                        # Obtener las coordenadas de los matches válidos
                        src_pts = np.float32([kp_f[m.queryIdx].pt for m in checked_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_m[m.trainIdx].pt for m in checked_matches]).reshape(-1, 1, 2)

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
                    except cv2.error as e:
                        print(f"Error al dibujar matches en la imagen {sufijos[i]}: {e}")
                else:
                    print(f"No hay suficientes matches válidos para la imagen {sufijos[i]}")
            else:
                print(f"No se encontraron matches válidos para la imagen {sufijos[i]}")

    plt.tight_layout()
    plt.show()


def imagenes_registradas(imagenes, sufijos, referencia, emparejamiento = "flann"):
    sift = cv2.SIFT_create()
# Detectar los puntos clave y calcular los descriptores para ambas imágenes
    kp_f, descrip_f = sift.detectAndCompute(referencia, mask=None)
    
    for i, imagen in enumerate(imagenes):
        if(i!=0):

            kp_m, descrip_m = sift.detectAndCompute(imagen, mask=None)

            if (emparejamiento == "flann"):
                FLANN_INDEX_KDTREE = 1  # Tipo de algoritmo a utilizar
                i_params = dict(algorithm=FLANN_INDEX_KDTREE , trees=5)  # Parámetros del indexador
                b_params = dict(checks=50)  # Número de verificaciones

                # Inicializar el matcher FLANN
                flann = cv2.FlannBasedMatcher(i_params, b_params)

                # Realizar el emparejamiento utilizando K-NN con k=2
                matches = flann.knnMatch(descrip_m, descrip_f, k=2)


                matches_filtrados = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        matches_filtrados.append(m)

            elif(emparejamiento == "ninguno"):
                #Usamos un matcheador a fuerza bruta si no usamos previamente algun metodo de matcheo como FLANN
                #o algun otro como KNN (cabe aclarar que esto igual sirve correctamente para ORB ya que la funcion
                # posee distancia de Hamming)

                #Todo esto se obtuvo de la documentacion de cv2

                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                #NORM_L2 seria eucladiano y crossCheck permite verificar que haya un match entre 
                #ambas imagenes segun sus puntos de interes
                #Usamos knn de manera default usando k=2, podriamos alterar este parametro
                matches = bf.match(descrip_m,descrip_f)
                #Hay que ordenar luego los matches (Esto fue tomado de la documentacion de cv2)
                matches = sorted(matches, key=lambda x: x.distance)

                matches_filtrados = [m for m in matches if m.distance < 0.7 * min([match.distance for match in matches])]

        

            coincidencias = cv2.drawMatches(imagen, kp_m,  referencia, kp_f, matches_filtrados, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            num_matches = min(len(kp_f), len(kp_m), len(matches_filtrados)) 
            #Obtenemos las coordenadas en donde se hicieron los matches 
            src_pts = np.float32([kp_f[m.trainIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_m[m.queryIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)

            print(len(src_pts))
            print(len(dst_pts))

            # Obtenemos la matriz homografica utilizando RANSAC para eliminar los outliers
            H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 5.0)
            # Obtener las dimensiones de la imagen 1
            h, w = referencia.shape
            imagen_registrada = cv2.warpPerspective(imagen, H, (w, h), flags=cv2.INTER_LINEAR)

            diferencia = cv2.absdiff(referencia, imagen_registrada)

            fig = plt.figure(figsize=(15, 10))  # Ajusta el tamaño de la figura
            

            # Primera fila: una imagen que ocupa todo el ancho
            ax1 = fig.add_subplot(2, 1, 1)  # 2 filas, 1 columna, primer subplot
            ax1.imshow(coincidencias, cmap='gray', vmin=0, vmax=255)
            ax1.set_title(f'Coincidencias con la imagen {sufijos[0]}')  # Cambia según necesites
            ax1.axis('off')

            # Segunda fila: tres imágenes
            ax2 = fig.add_subplot(2, 3, 4)  # 2 filas, 3 columnas, cuarto subplot
            ax2.imshow(imagen, cmap='gray', vmin=0, vmax=255)  # Imagen original
            ax2.set_title('Imagen Original')
            ax2.axis('off')

            ax3 = fig.add_subplot(2, 3, 5)  # 2 filas, 3 columnas, quinto subplot
            ax3.imshow(imagen_registrada, cmap='gray', vmin=0, vmax=255)
            ax3.set_title(f'Registro con {sufijos[1]}')  # Cambia según necesites
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 3, 6)  # 2 filas, 3 columnas, sexto subplot
            ax4.imshow(diferencia, cmap='gray')
            ax4.set_title('Diferencia entre Imágenes')
            ax4.axis('off')

            plt.tight_layout(pad=3.0)  # Ajusta el espacio entre subgráficas
            plt.show()