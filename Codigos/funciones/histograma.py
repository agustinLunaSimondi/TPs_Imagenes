import  matplotlib.pyplot as plt
import numpy as np
def histograma(imagenes,sufijos):
  plt.figure(figsize=(22, 8))
  for i, imagen in enumerate(imagenes):
    plt.subplot(1, 5, i + 1)  # Define la ubicación del subplot
    plt.hist(np.ravel(imagen), bins=256, range=(0, 255))
    plt.title(sufijos[i])