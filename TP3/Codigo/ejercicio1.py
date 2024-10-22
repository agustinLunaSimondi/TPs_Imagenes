from funciones.cargar import cargar_imagenes, mostrar_imagenes

def ejercicio1():
    # Set 1: Imágenes de la carpeta "PAIByB-3"
    sufijos_set1 = ['img-1', 'img-2', 'img-3', 'img-4','img-11','img-12']
    ruta_base_set1 = './TP3/Imagenes/PAIByB-5/'  # Ruta base para el set 1


    
    # Cargar imágenes del set 1
    imagenes_set1 = cargar_imagenes(sufijos_set1, ruta_base_set1)
    mostrar_imagenes(imagenes_set1,sufijos_set1)
   


  

if __name__ == "__main__":
    ejercicio1()