import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calcular_estadisticas(imagenes, sufijos):
  for imagen in range(len(imagenes)):
    print("El valor esperado de "+sufijos[imagen]+".tif es:", np.mean(imagenes[imagen]))
    print("La varianza de "+sufijos[imagen]+".tif es:", np.var(imagenes[imagen]))
    print()

def histograma(imagenes,sufijos):
  plt.figure(figsize=(22, 8))
  for i, imagen in enumerate(imagenes):
    plt.subplot(1, len(imagenes), i + 1)  # Define la ubicación del subplot
    plt.hist(np.ravel(imagen), bins=256, range=(0, 255))
    plt.title(sufijos[i])
  plt.show()

def comparacion (imagenes,sufijos):
    img_ref = imagenes[0]
    for i in range(1, len(imagenes)):
        img_registrada = imagenes[i]
        mse = np.mean((img_ref - img_registrada) ** 2) 
        print(f"{sufijos[i]}-Error Cuadrático Medio (MSE): {mse:.4f}")
        ssim_index, _ = ssim(img_ref, img_registrada, full=True,  data_range=1.0)
        print(f"{sufijos[i]}-Índice de Similitud Estructural (SSIM): {ssim_index:.4f}")  
