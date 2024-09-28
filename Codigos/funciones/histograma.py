import cv2
import matplotlib.pyplot as plt
import numpy as np

def mostrar_histogramas(imagenes, sufijos):
    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
        else:
            # Verificar el número de canales de la imagen
            if len(imagen.shape) == 3:  # Imagen con 3 canales (color)
                plt.figure(figsize=(15, 5))

                # Espacio de color RGB con histograma combinado
                plt.subplot(1, 3, 1)
                plt.title(f"Histograma RGB - {sufijos[i]}")
                colores = ('r', 'g', 'b')  # Rojo, Verde, Azul (RGB)
                nombres_canales = ['R', 'G', 'B']
                hist_total = np.zeros(256)  # Inicializar el histograma combinado como un vector 1D

                for j, col in enumerate(colores):
                    hist = cv2.calcHist([imagen], [j], None, [256], [0, 256])
                    plt.plot(hist, color=col, label=nombres_canales[j])  # Dibujar cada canal con su color
                    hist_total += hist.flatten()  # Sumar al histograma combinado

                # Dibujar el histograma combinado en color gris
                plt.plot(hist_total, color='gray', label='RGB combinado', linestyle='--', linewidth=1.5)
                plt.xlim([0, 256])
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
                plt.legend()  # Mostrar la leyenda para los canales

                # Espacio de color YUV
                yuv_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)
                plt.subplot(1, 3, 2)
                plt.title(f"Histograma YUV - {sufijos[i]}")
                componentes_yuv = ('Y', 'U', 'V')
                for j, comp in enumerate(componentes_yuv):
                    hist = cv2.calcHist([yuv_img], [j], None, [256], [0, 256])
                    plt.plot(hist, label=comp)
                plt.xlim([0, 256])
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
                plt.legend()

                # Espacio de color HSV
                hsv_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                plt.subplot(1, 3, 3)
                plt.title(f"Histograma HSV - {sufijos[i]}")
                componentes_hsv = ('H', 'S', 'V')
                for j, comp in enumerate(componentes_hsv):
                    hist = cv2.calcHist([hsv_img], [j], None, [256], [0, 256])
                    plt.plot(hist, label=comp)
                plt.xlim([0, 256])
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')
                plt.legend()

            else:  # Imagen en escala de grises (1 canal)
                plt.figure(figsize=(5, 5))
                plt.title(f"Histograma Grayscale - {sufijos[i]}")
                hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
                plt.xlim([0, 256])
                plt.xlabel('Intensidad')
                plt.ylabel('Frecuencia')

            plt.tight_layout()
            plt.show()