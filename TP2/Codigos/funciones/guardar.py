import os
import cv2

def guardar_imagen(path, nombre, img, color=True):
    """Guardar imagen usando OpenCV."""
    if color:
        cv2.imwrite(os.path.join(path, nombre), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(os.path.join(path, nombre), img)