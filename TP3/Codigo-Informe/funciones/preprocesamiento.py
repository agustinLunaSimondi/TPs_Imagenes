import cv2
def redimensionar(imagenes, indices_redimensionar):
    """
    Redimensiona las imágenes en el set que se encuentran en los índices especificados
    para que tengan el mismo tamaño que la primera imagen del set (imagen de referencia).
    
    Args:
        imagenes: Lista de imágenes de entrada en formato BGR.
        indices_redimensionar: Lista de índices que indican qué imágenes deben ser redimensionadas.
        
    Returns:
        Lista de imágenes donde las imágenes en los índices especificados han sido redimensionadas.
    """
    # Obtener las dimensiones de la primera imagen (de referencia)
    referencia_alto, referencia_ancho = imagenes[0].shape[:2]
    
    # Copiar el set original para no modificar el set original directamente
    imagenes_modificadas = imagenes.copy()
    
    # Redimensionar solo las imágenes en los índices proporcionados
    for i in indices_redimensionar:
        imagenes_modificadas[i] = cv2.resize(imagenes_modificadas[i], (referencia_ancho, referencia_alto))
    
    return imagenes_modificadas