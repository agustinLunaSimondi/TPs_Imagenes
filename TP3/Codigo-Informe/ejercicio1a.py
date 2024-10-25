from funciones.cargar import cargar_imagenes, mostrar_imagenes, analizar_imagenes
from funciones.metricas import calcular_estadisticas, histograma
from funciones.poi import obtener_caracteristicas, imagenes_registradas
from funciones.preprocesamiento import redimensionar
def ejercicio1a():
    # Set 1: Imágenes de la carpeta "PAIByB-3"
    sufijos_set1 = ['img-1', 'img-2', 'img-3', 'img-4','img-11','img-12']
    ruta_base_set1 = './TP3/Imagenes/PAIByB-5/'  # Ruta base para el set 1


    
    # Cargar imágenes del set 1
    imagenes_set1 = cargar_imagenes(sufijos_set1, ruta_base_set1)
    mostrar_imagenes(imagenes_set1,sufijos_set1)
    
    #analizar_imagenes (imagenes_set1,sufijos_set1)
    #calcular_estadisticas(imagenes_set1,sufijos_set1)
    #histograma(imagenes_set1,sufijos_set1)
    print("Al ver imagenes de diferentes dimensiones hacemos un resize tomando la imagen de referencia como la primera")
    indices = [4,5]
    imagenes_set1 = redimensionar(imagenes_set1,indices)
    #mostrar_imagenes(imagenes_set1,sufijos_set1)
    print("Se hace analisis de esta parte primero para entender las diferencias")
    print("Hacemos deteccion de las caracteristicas con SIFT")
    #obtener_caracteristicas(imagenes_set1,sufijos_set1,metodo ='SIFT')
    print("Ahora se lleva acabo el caso para ORB")
    #obtener_caracteristicas(imagenes_set1,sufijos_set1,metodo ='ORB')

    print("Una vez hecho esto ahora vemos como se unen los puntos y su registracion utilizando ningun filtrado solamente distancai eucladiana")

    imagenes_registradas(imagenes_set1,sufijos_set1,imagenes_set1[0],emparejamiento = "flann")

if __name__ == "__main__":
    ejercicio1a()