from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.poi import comparar_metodos_puntos_interes

def ejercicio3a():
# Set 2: Imágenes de la carpeta "PAIByB-4"
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'  # Ruta base para el set 

    

    # Cargar imágenes del set 2
    imagenes_set2 = cargar_imagenes(sufijos_set2, ruta_base_set2)
    comparar_metodos_puntos_interes(imagenes_set2, sufijos_set2)

if __name__ == "__main__":
    ejercicio3a()