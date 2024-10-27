import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calcular_estadisticas(imagenes, sufijos):
  for imagen in range(len(imagenes)):
    print("El valor esperado de "+sufijos[imagen]+".tif es:", np.mean(imagenes[imagen]))
    print("La varianza de "+sufijos[imagen]+".tif es:", np.var(imagenes[imagen]))
    print()

def histograma(imagenes, sufijos, ymax=None):
    plt.figure(figsize=(22, 8))
    for i, imagen in enumerate(imagenes):
        plt.subplot(1, len(imagenes), i + 1)
        plt.hist(np.ravel(imagen), bins=256, range=(0, 255))
        if ymax:
            plt.ylim(0, ymax)  # Limita el rango del eje y a ymax
        plt.title(sufijos[i])
    plt.show()

def comparacion(imagenes,sufijos):
  referencia = imagenes[0]
  for i in range(1,len(imagenes)):
    imagen_movil = imagenes[i]
    mse = np.mean((referencia - imagen_movil) ** 2)
    ssim_index, _ = ssim(referencia, imagen_movil, full=True, data_range=1.0)
    print(f"Imagen {sufijos[i]} - Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"Imagen {sufijos[i]} - Índice de Similitud Estructural (SSIM): {ssim_index:.4f}")
    
  