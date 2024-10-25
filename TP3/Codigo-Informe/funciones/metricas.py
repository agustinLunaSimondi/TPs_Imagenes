import numpy as np
import matplotlib.pyplot as plt

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

  