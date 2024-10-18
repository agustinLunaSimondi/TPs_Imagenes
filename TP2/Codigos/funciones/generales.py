import numpy as np
import matplotlib.pyplot as plt

def plot_1(imagen, t = "Imagen", color = 'gray'):
    plt.figure(figsize=(7, 7))
    plt.imshow(imagen, vmin=0, vmax=255, cmap = color)
    plt.title(t, fontsize=15)
    plt.axis("off")
    plt.show()
    return


def plot_2(imagen1, imagen2, t1 = "Imagen 1", t2 = "Imagen 2", color = 'gray'):
    fig,ax = plt.subplots(1, 2, figsize=(15,20))
    ax[0].imshow(imagen1, vmin=0, vmax=255, cmap = color)
    ax[0].set_title(t1, fontsize=15)
    ax[1].imshow(imagen2, vmin=0, vmax=255, cmap = color)
    ax[1].set_title(t2, fontsize=15)
    plt.show()
    return