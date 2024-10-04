from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.histograma import mostrar_histogramas, mostrar_espacios
from funciones.autocorrelograma import autocorrelograma_por_canal , autocorrelograma_rgb
from funciones.guardar import guardar_imagen

def ejercicio1():
    # Set 1: Imágenes de la carpeta "PAIByB-3"
    sufijos_set1 = ['Image-1', 'Image-2', 'Image-3', 'Image-4']
    ruta_base_set1 = './Imagenes/PAIByB-3/'  # Ruta base para el set 1


    
    # Cargar imágenes del set 1
    imagenes_set1 = cargar_imagenes(sufijos_set1, ruta_base_set1)
   
    # Mostrar histogramas del set 1
    print("Mostrando histogramas del Set 1 (PAIByB-3):")
    mostrar_espacios(imagenes_set1, sufijos_set1)
    mostrar_histogramas(imagenes_set1, sufijos_set1)

    print("Mostrando autocorrelogramas del Set 1 (PAIByB-3):")
    #autocorrelograma_por_canal(imagenes_set1,sufijos_set1)
    #autocorrelograma_rgb(imagenes_set1,sufijos_set1)

  

if __name__ == "__main__":
    ejercicio1()
