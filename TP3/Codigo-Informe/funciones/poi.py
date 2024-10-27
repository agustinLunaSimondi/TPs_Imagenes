import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from funciones.metricas import comparacion


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




def imagenes_registradas(imagenes, sufijos, referencia,metodo ="sift" ,emparejamiento = "flann",lowe = True, ransac = True):
    if(metodo == "sift"):
        met = cv2.SIFT_create()
    if(metodo == "orb"):
        met = cv2.ORB_create()
# Detectar los puntos clave y calcular los descriptores para ambas imágenes
    kp_f, descrip_f = met.detectAndCompute(referencia, mask=None)
    registradas=[]
    for i, imagen in enumerate(imagenes):
        if(i!=0):
            print(i)
            kp_m, descrip_m = met.detectAndCompute(imagen, mask=None)

            if (emparejamiento == "flann"):
                if(metodo == "sift"):
                    FLANN_INDEX_KDTREE = 1  # Tipo de algoritmo a utilizar
                    i_params = dict(algorithm=FLANN_INDEX_KDTREE , trees=5)  # Parámetros del indexador
                    b_params = dict(checks=50)  # Número de verificaciones

                if(metodo == "orb"):
                    FLANN_INDEX_LSH = 6  # Tipo de algoritmo a utilizar
                    i_params = dict(algorithm=FLANN_INDEX_LSH , table_number=6,
                          key_size = 12, multi_probe_level = 1          
                                    )  # Parámetros del indexador
                    b_params = dict(checks=50)  # Número de verificaciones
                # Inicializar el matcher FLANN
                flann = cv2.FlannBasedMatcher(i_params, b_params)

                # Realizar el emparejamiento utilizando K-NN con k=2
                matches = flann.knnMatch(descrip_m, descrip_f, k=2)


                
            elif(emparejamiento == "knn" and metodo == "sift"):
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(descrip_m,descrip_f,k=2)
 
            
                
            elif(emparejamiento == None):
                #Usamos un matcheador a fuerza bruta si no usamos previamente algun metodo de matcheo como FLANN
                

                #Todo esto se obtuvo de la documentacion de cv2
                if(metodo == "sift"):
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                #NORM_L2 seria eucladiano y crossCheck permite verificar que haya un match entre 
                #ambas imagenes segun sus puntos de interes
                elif( metodo == "orb"):
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
                matches = bf.match(descrip_m,descrip_f)
                #Hay que ordenar luego los matches (Esto fue tomado de la documentacion de cv2)
                matches_filtrados = sorted(matches, key=lambda x: x.distance)



            if (lowe and emparejamiento != None):
                matches_filtrados = []
                for m, n in matches:
                    if m.distance < 0.70 * n.distance:
                        matches_filtrados.append(m)
            else:
                # Para el caso de emparejamiento a fuerza bruta
                if emparejamiento == "flann" or emparejamiento == "knn":
                    matches_filtrados = [m for m, n in matches]  # Solo la primera coincidencia de cada par
                else:
                    matches_filtrados = matches  # Si es None, asigna matches directamente 
                    

            coincidencias = cv2.drawMatches(imagen, kp_m,  referencia, kp_f, matches_filtrados, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
           


            #Obtenemos las coordenadas en donde se hicieron los matches 
            src_pts = np.float32([kp_f[m.trainIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_m[m.queryIdx].pt for m in matches_filtrados]).reshape(-1, 1, 2)

        

            # Obtenemos la matriz homografica utilizando RANSAC para eliminar los outliers
            if(ransac):
                H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 5.0)
            else:
                H, mask = cv2.findHomography(dst_pts,src_pts)
            # Obtener las dimensiones de la imagen 1
            h, w = referencia.shape
            imagen_registrada = cv2.warpPerspective(imagen, H, (w, h), flags=cv2.INTER_LINEAR)

            diferencia = cv2.absdiff(referencia, imagen_registrada)

            fig = plt.figure(figsize=(15, 10))  # Ajusta el tamaño de la figura
            
            plt.suptitle(f'Método: {metodo}, Emparejamiento: {emparejamiento}, Lowe: {lowe}, RANSAC: {ransac}', fontsize=16)

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
            ax3.set_title(f'Registro con {sufijos[i]}')  # Cambia según necesites
            ax3.axis('off')

            ax4 = fig.add_subplot(2, 3, 6)  # 2 filas, 3 columnas, sexto subplot
            ax4.imshow(diferencia, cmap='gray')
            ax4.set_title('Diferencia entre Imágenes')
            ax4.axis('off')

            plt.tight_layout(pad=3.0)  # Ajusta el espacio entre subgráficas
            plt.show()
            registradas.append(imagen_registrada)

            comparacion([referencia,imagen_registrada],["blank",sufijos[i]])
    return registradas