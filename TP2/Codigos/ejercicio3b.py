from funciones.cargar import cargar_imagenes, mostrar_imagenes
from funciones.evaluar_transformaciones import evaluar_transformaciones

def ejercicio3b():
# Set 2: Imágenes de la carpeta "PAIByB-4"
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'  # Ruta base para el set 

    

    # Cargar imágenes del set 2
    imagenes_set2 = cargar_imagenes(sufijos_set2, ruta_base_set2)
    
    for i, imagen in enumerate(imagenes_set2):
        evaluar_transformaciones(imagen, sufijos_set2[i])
if __name__ == "__main__":
    ejercicio3b()