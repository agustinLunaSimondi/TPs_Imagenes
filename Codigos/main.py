from funciones.cargar import cargar_imagenes, mostrar_imagenes

def main():
    # Set 1: Imágenes de la carpeta "PAIByB-3"
    sufijos_set1 = ['Image-1', 'Image-2', 'Image-3', 'Image-4']
    ruta_base_set1 = './Imagenes/PAIByB-3/'  # Ruta base para el set 1

    # Set 2: Imágenes de la carpeta "PAIByB-4"
    sufijos_set2 = ['img-1', 'img-2', 'img-3', 'img-4', 'img-5']
    ruta_base_set2 = './Imagenes/PAIByB-4/'  # Ruta base para el set 2

    # Cargar imágenes del set 1
    imagenes_set1 = cargar_imagenes(sufijos_set1, ruta_base_set1)

    # Cargar imágenes del set 2
    imagenes_set2 = cargar_imagenes(sufijos_set2, ruta_base_set2)

    # Mostrar imágenes del set 1
    print("Mostrando imágenes del Set 1 (PAIByB-3):")
    mostrar_imagenes(imagenes_set1, sufijos_set1, barra=True)

    # Mostrar imágenes del set 2
    print("Mostrando imágenes del Set 2 (PAIByB-4):")
    mostrar_imagenes(imagenes_set2, sufijos_set2, barra=True)

if __name__ == "__main__":
    main()