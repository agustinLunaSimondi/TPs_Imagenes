import cv2
import matplotlib.pyplot as plt
import numpy as np

def cargar_imagenes(sufijos, ruta_base):
    imagenes_descargadas = [None] * len(sufijos)

    for i, sufijo in enumerate(sufijos):
        imagen_path = ruta_base+sufijo+'.tif'

        imagenes_descargadas[i] = cv2.imread(imagen_path, cv2.IMREAD_COLOR)

        if imagenes_descargadas[i] is None:
          #Si por algun motivo no  carga que lo informe
            print(f"Error: No se cargó correctamente el path de la imagen {sufijo}.tif")
        else:
            print(f"Se cargó correctamente el path de la imagen {sufijo}.tif")

    return imagenes_descargadas

def mostrar_imagenes(imagenes, sufijos, barra = False):

    plt.figure(figsize=(20, 10))

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")

        else:

            if len(np.shape(imagen)) == 1: #Esto lo hago en casod e ser una linea
            # Crear subtítulos y mostrar la imagen
              linea= True
              plt.subplot(1, len(imagenes), i + 1)
              plt.title(f'{sufijos[i]}')
              plt.plot(imagen)
            else:
            # Crear subtítulos y mostrar la imagen
              linea= False
              plt.subplot(1, len(imagenes)+1, i + 1)
              plt.title(f'{sufijos[i]}')
              plt.imshow(imagen, cmap='gray',vmin=0,vmax=255)
              plt.axis('on')
              plt.tight_layout()

    #Pongo la barra de intensidades solo si no es una linea
    if (not linea and barra):
      clb = plt.colorbar(shrink=0.25)
      clb.set_label('Niveles de grises', fontsize=12)

    plt.show()


def analizar_imagenes(imagenes, sufijos):
    altura_max = 0
    ancho_max = 0

    for i, imagen in enumerate(imagenes):
        if imagen is None:
            print(f"Error: No se cargó la imagen {sufijos[i]}.tif")
        else:
            altura, ancho = imagen.shape
            maximo = np.max(imagen)
            minimo = np.min(imagen)
            print(f"La altura y ancho de la imagen {sufijos[i]}.tif son: {altura} y {ancho}. Máximo: {maximo}, Mínimo: {minimo}")

    """ Por mas de que parezca una obviedad analizamos las dimensiones de todas las imagenes
    #para asegurarnos que miden lo mismo y tomamos su altura y ancho para que en el inciso I del Ej 1
    # No nos pasemos de pixeles cuando se lo pedimos al usuario"""
    return altura, ancho