from funciones.cargar import cargar_imagenes, mostrar_imagenes, analizar_imagenes
from funciones.metricas import *

def ejercicio1():
    # Set 1: Imágenes de la carpeta "PAIByB-6"
    sufijos_set2 = ['img-1', 'img-2', 'img-3','img-11','img-12']
    ruta_base_set2 = './TP3/Imagenes/PAIByB-6/'  # Ruta base para el set 1


    
    # Cargar imágenes del set 1
    imagenes_set2 = cargar_imagenes(sufijos_set2, ruta_base_set2)
    mostrar_imagenes(imagenes_set2,sufijos_set2)
   
   
    analizar_imagenes (imagenes_set2,sufijos_set2)
    calcular_estadisticas(imagenes_set2,sufijos_set2)
    histograma(imagenes_set2,sufijos_set2)

  

if __name__ == "__main__":
    ejercicio1()