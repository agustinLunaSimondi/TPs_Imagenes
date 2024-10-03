import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def rotar_imagen(imagen, angulo):
    """Rota la imagen dado un ángulo en grados."""
    (h, w) = imagen.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, M, (w, h))
    return imagen_rotada

def escalar_imagen(imagen, factor):
    """Escala la imagen dado un factor."""
    (h, w) = imagen.shape[:2]
    nueva_dim = (int(w * factor), int(h * factor))
    imagen_escalada = cv2.resize(imagen, nueva_dim, interpolation=cv2.INTER_LINEAR)
    return imagen_escalada

def ajustar_brillo_contraste(imagen, alfa=1.0, beta=0):
    """
    Ajusta el brillo y contraste de la imagen.
    alfa: Factor de contraste.
    beta: Factor de brillo.
    """
    return cv2.convertScaleAbs(imagen, alpha=alfa, beta=beta)

def detectar_puntos_interes(imagen, metodo='SIFT'):
    """
    Detecta puntos de interés en una imagen usando SIFT u ORB.
    
    Args:
        imagen: La imagen de entrada en formato BGR.
        metodo: El método a usar para detectar puntos de interés ('SIFT', 'ORB').
    
    Returns:
        keypoints: Lista de puntos clave detectados.
        tiempo: Tiempo de ejecución del método.
    """
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

    inicio_tiempo = time.time()  # Iniciar temporizador

    if metodo == 'SIFT':
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    elif metodo == 'ORB':
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)

    tiempo = time.time() - inicio_tiempo  # Calcular tiempo de ejecución
    output_image = cv2.drawKeypoints(imagen, keypoints, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return keypoints,output_image, tiempo

def evaluar_transformaciones(imagen, sufijo, metodos=['SIFT', 'ORB'], output_dir='./output/transformaciones'):
    """
    Aplica diversas transformaciones a la imagen y evalúa los métodos de detección de puntos clave.
    
    Args:
        imagen: Imagen en formato BGR.
        sufijo: Identificador de la imagen.
        metodos: Lista de métodos para evaluar ('SIFT', 'ORB').
        output_dir: Directorio donde guardar los resultados.
    """
    os.makedirs(output_dir, exist_ok=True)  # Crear directorio de salida si no existe
    
    transformaciones = {
        'Rotación 30°': lambda img: rotar_imagen(img, 30),
        'Rotación 90°': lambda img: rotar_imagen(img, 90),
        'Escala 0.5': lambda img: escalar_imagen(img, 0.5),
        'Escala 1.5': lambda img: escalar_imagen(img, 1.5),
        'Ajuste Brillo (alfa=1.2, beta=30)': lambda img: ajustar_brillo_contraste(img, alfa=1.2, beta=30),
        'Ajuste Contraste (alfa=1.5, beta=-30)': lambda img: ajustar_brillo_contraste(img, alfa=1.5, beta=-30)
    }

    # Aplicar cada transformación y evaluar los métodos
    for nombre_transformacion, transformacion in transformaciones.items():
        imagen_transformada = transformacion(imagen)

        print(f"Evaluando transformación: {nombre_transformacion} en {sufijo}")

        for metodo in metodos:
            
            keypoints,output_image,tiempo = detectar_puntos_interes(imagen_transformada, metodo=metodo)
            num_puntos = len(keypoints)

            print(f"  {metodo}: {num_puntos} puntos clave detectados, Tiempo = {tiempo:.4f} segundos")

            # Guardar la imagen con los puntos clave dibujados
            output_image = cv2.drawKeypoints(imagen_transformada, keypoints, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
            output_path = os.path.join(output_dir, f"{sufijo}_{nombre_transformacion.replace(' ', '_')}_{metodo}.png")
            cv2.imwrite(output_path, output_image)
        if sufijo == 'img-1' : 
            for j, metodo in enumerate(metodos):
                keypoints,output_image,tiempo = detectar_puntos_interes(imagen_transformada, metodo=metodo)
                try:
                    # Detectar puntos de interés con el método especificado
                    
                    
                    # Número de puntos clave detectados
                    
                    
                    # Mostrar la imagen con los puntos de interés
                    plt.subplot(1, len(metodos), j + 1)
                    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
                    plt.suptitle(f'Evaluando transformación: {nombre_transformacion} en {sufijo}')
                    plt.title(f"{metodo} - {num_puntos} puntos\n{tiempo:.4f} seg")
                    plt.axis('off')
                    
                    

                except cv2.error as e:
                    print(f"Error usando {metodo} en {sufijo}: {e}")

            plt.tight_layout()
            plt.show()
