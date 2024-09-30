import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def guardar_imagen(path, nombre, img, color=True):
    """Guardar imagen usando OpenCV."""
    if color:
        cv2.imwrite(os.path.join(path, nombre), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(os.path.join(path, nombre), img)

def mostrar_histogramas(imagenes, sufijos, output_dir='./output'):
    os.makedirs(output_dir, exist_ok=True)  # Crear el directorio principal si no existe

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
        else:
            # Crear directorio para la imagen
            imagen_dir = os.path.join(output_dir, sufijos[i])
            os.makedirs(imagen_dir, exist_ok=True)

            if len(imagen.shape) == 3:  # Imagen en color (3 canales)
                # Crear subcarpetas para RGB, YUV y HSV
                rgb_dir = os.path.join(imagen_dir, 'RGB')
                yuv_dir = os.path.join(imagen_dir, 'YUV')
                hsv_dir = os.path.join(imagen_dir, 'HSV')
                os.makedirs(rgb_dir, exist_ok=True)
                os.makedirs(yuv_dir, exist_ok=True)
                os.makedirs(hsv_dir, exist_ok=True)

                ### Espacio de color RGB ###
                plt.figure(figsize=(20, 15))

                # Guardar y mostrar imágenes de cada componente (R, G, B)
                colores = ('R', 'G', 'B')
                for j, col in enumerate(colores):
                    canal_img = np.zeros_like(imagen)
                    canal_img[:, :, j] = imagen[:, :, j]
                    
                    # Guardar cada canal individual en la carpeta RGB
                    canal_img_rgb = cv2.cvtColor(canal_img, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para guardar
                    guardar_imagen(rgb_dir, f"{sufijos[i]}_Canal_{col}.png", canal_img_rgb)

                    # Mostrar cada canal
                    plt.subplot(3, 4, 1 + j)
                    plt.imshow(canal_img_rgb)
                    plt.title(f"Canal {col} - {sufijos[i]}")
                    plt.axis('off')

                # Guardar el histograma RGB combinado y por canal
                plt.subplot(3, 4, 4)
                plt.title(f"Histograma RGB - {sufijos[i]}")
                colores_rgb = ('r', 'g', 'b')
                hist_total = np.zeros(256) 
                for j, col in enumerate(colores_rgb):
                    hist = cv2.calcHist([imagen], [j], None, [256], [0, 256])
                    hist_total += hist.flatten() 
                    plt.plot(hist, color=col, label=f'Canal {col.upper()}')
                plt.plot(hist_total, color='gray', label='RGB combinado', linestyle='--', linewidth=1.5)
                plt.xlim([0, 256])
                plt.legend()

                # Guardar el histograma como imagen en la carpeta RGB
                plt.savefig(os.path.join(rgb_dir, f"{sufijos[i]}_Histograma_RGB.png"))

                ### Espacio de color YUV ###
                # Convertir la imagen a YUV y guardar
                yuv_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)
                componentes_yuv = ('Y', 'U', 'V')
                for j in range(3):
                    canal_img = np.zeros_like(yuv_img)
                    canal_img[:, :, j] = yuv_img[:, :, j]

                    # Guardar cada canal Y, U, V en la carpeta YUV
                    guardar_imagen(yuv_dir, f"{sufijos[i]}_Canal_{componentes_yuv[j]}.png", canal_img[:, :, j], color=False)

                    # Mostrar cada canal
                    plt.subplot(3, 4, 5 + j)
                    plt.imshow(canal_img[:, :, j], cmap='gray')
                    plt.title(f"Canal {componentes_yuv[j]} - {sufijos[i]}")
                    plt.axis('off')

                # Guardar el histograma YUV
                plt.subplot(3, 4, 8)
                plt.title(f"Histograma YUV - {sufijos[i]}")
                hist_total = np.zeros(256) 
                for j, comp in enumerate(componentes_yuv):
                    hist = cv2.calcHist([yuv_img], [j], None, [256], [0, 256])
                    hist_total += hist.flatten() 
                    plt.plot(hist, label=f'Canal {comp}')
                plt.plot(hist_total, color='gray', label='RGB combinado', linestyle='--', linewidth=1.5)
                plt.xlim([0, 256])
                plt.legend()

                # Guardar el histograma como imagen en la carpeta YUV
                plt.savefig(os.path.join(yuv_dir, f"{sufijos[i]}_Histograma_YUV.png"))

                ### Espacio de color HSV ###
                # Convertir la imagen a HSV y guardar
                hsv_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
                componentes_hsv = ('H', 'S', 'V')
                for j in range(3):
                    canal_img = np.zeros_like(hsv_img)
                    canal_img[:, :, j] = hsv_img[:, :, j]

                    # Guardar cada canal H, S, V en la carpeta HSV
                    guardar_imagen(hsv_dir, f"{sufijos[i]}_Canal_{componentes_hsv[j]}.png", canal_img[:, :, j], color=False)

                    # Mostrar cada canal
                    plt.subplot(3, 4, 9 + j)
                    plt.imshow(canal_img[:, :, j], cmap='gray')
                    plt.title(f"Canal {componentes_hsv[j]} - {sufijos[i]}")
                    plt.axis('off')

                # Guardar el histograma HSV
                plt.subplot(3, 4, 12)
                plt.title(f"Histograma HSV - {sufijos[i]}")
                hist_total = np.zeros(256) 
                for j, comp in enumerate(componentes_hsv):
                    hist = cv2.calcHist([hsv_img], [j], None, [256], [0, 256])
                    hist_total += hist.flatten() 
                    plt.plot(hist, label=f'Canal {comp}')
                plt.plot(hist_total, color='gray', label='RGB combinado', linestyle='--', linewidth=1.5)
                plt.xlim([0, 256])
                plt.legend()

                # Guardar el histograma como imagen en la carpeta HSV
                plt.savefig(os.path.join(hsv_dir, f"{sufijos[i]}_Histograma_HSV.png"))

            else:  # Imagen en escala de grises (1 canal)
                grayscale_dir = os.path.join(imagen_dir, 'Grayscale')
                os.makedirs(grayscale_dir, exist_ok=True)

                # Guardar la imagen en escala de grises
                guardar_imagen(grayscale_dir, f"{sufijos[i]}_Grayscale.png", imagen, color=False)

                # Mostrar la imagen en escala de grises
                plt.subplot(1, 2, 1)
                plt.imshow(imagen, cmap='gray')
                plt.title(f"Imagen Grayscale - {sufijos[i]}")
                plt.axis('off')

                # Mostrar el histograma de la imagen en escala de grises
                plt.subplot(1, 2, 2)
                plt.title(f"Histograma Grayscale - {sufijos[i]}")
                hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
                plt.xlim([0, 256])

                # Guardar el histograma como imagen en la carpeta Grayscale
                plt.savefig(os.path.join(grayscale_dir, f"{sufijos[i]}_Histograma_Grayscale.png"))
            plt.subplots_adjust(hspace=0.3,wspace= 0.3)
            
            #plt.tight_layout()
            plt.show()