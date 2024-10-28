import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform, img_as_float
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from scipy.optimize import differential_evolution

def fondo_blanco_a_negro(img, I_threshold = 240):
    imagen = img.copy()
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagen

    _, mask = cv2.threshold(gray, I_threshold, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    imagen[mask == 255] = 0
    return imagen


def invertir(imagen):
    if len(imagen.shape) == 3:
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagen
    imagen_invertida = 255 - gray
    return imagen_invertida


def registrar_MI(img_ref, img_mov):
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if len(img_ref.shape) == 3 else img_ref
    img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY) if len(img_mov.shape) == 3 else img_mov

    img_mov = cv2.resize(img_mov, (img_ref.shape[1], img_ref.shape[0]))

    hgram, x_edges, y_edges = np.histogram2d(img_ref.ravel(), img_mov.ravel(), bins=256)
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    non_zero = pxy > 0

    mi_initial = np.sum(pxy[non_zero] * np.log(pxy[non_zero] / px_py[non_zero]))
    print(f"Información Mutua inicial: {mi_initial:.6f}")

    initial_params = [1, 0, 0, 1, 0, 0]

    def cost_function(params):
        a, b, c, d, e, f = params
        # Construir la matriz de transformación afín
        tform_matrix = np.array([[a, b, e],
                                [c, d, f],
                                [0, 0, 1]])

        # Crear el objeto AffineTransform
        tform = transform.AffineTransform(matrix=tform_matrix)

        # Aplicar la transformación inversa a la imagen movida
        img_mov_transformed = transform.warp(img_mov, tform.inverse, order=3)  # Interpolación cúbica

        # Calcular el histograma conjunto entre la imagen de referencia y la imagen transformada
        hgram_transformed, _, _ = np.histogram2d(img_ref.ravel(), img_mov_transformed.ravel(), bins=256)

        # Convertir el histograma conjunto a probabilidades
        pxy_transformed = hgram_transformed / float(np.sum(hgram_transformed))
        px_transformed = np.sum(pxy_transformed, axis=1)  # Marginal de img_ref
        py_transformed = np.sum(pxy_transformed, axis=0)  # Marginal de img_mov_transformed

        # Evitar log(0) estableciendo valores mínimos
        px_py_transformed = px_transformed[:, None] * py_transformed[None, :]
        non_zero_transformed = pxy_transformed > 0  # Ignorar entradas cero

        # Calcular la Información Mutua
        mi_transformed = np.sum(pxy_transformed[non_zero_transformed] * np.log(pxy_transformed[non_zero_transformed] / px_py_transformed[non_zero_transformed]))

        # Retornar el negativo de la MI porque 'minimize' busca minimizar
        return -mi_transformed

    bounds_affine = [
        (0.1, 1.0),    # a: escalado en X
        (-0.7, 0.7),   # b: rotación/cizallamiento en X
        (-0.7, 0.7),   # c: rotación/cizallamiento en Y
        (0.9, 1.0),    # d: escalado en Y
        (-20, 20),     # e: traslación en X
        (-20, 20)      # f: traslación en Y
    ]

    opt_options = {
        'maxiter': 1000,
        'disp': True  # Mostrar información de la optimización
    }

    result = differential_evolution(
        cost_function,
        bounds_affine,
        maxiter=1000,
        disp=True
    )

    if not result.success:
        raise ValueError("La optimización no convergió: " + result.message)
    
    optimal_params = result.x
    print(f"Parámetros óptimos: a={optimal_params[0]:.6f}, b={optimal_params[1]:.6f}, c={optimal_params[2]:.6f}, d={optimal_params[3]:.6f}, e={optimal_params[4]:.2f}, f={optimal_params[5]:.2f}")

    optimal_tform_matrix = np.array([[optimal_params[0], optimal_params[1], optimal_params[4]],
                                     [optimal_params[2], optimal_params[3], optimal_params[5]],
                                     [0,                  0,                 1          ]])
    
    optimal_tform = transform.AffineTransform(matrix=optimal_tform_matrix)
    aligned_img = transform.warp(img_mov, optimal_tform.inverse, order=3)  # Interpolación cúbica

    mse_val = np.mean((img_ref - aligned_img) ** 2)
    print(f"Error Cuadrático Medio (MSE): {mse_val:.6f}")

    ssim_val, _ = ssim(img_ref, aligned_img, full=True, data_range=1.0)
    print(f"Índice de Similitud Estructural (SSIM): {ssim_val:.6f}")


    # Visualizar las imágenes: referencia, original y alineada
    plt.figure(figsize=(18, 6))

    # Imagen de referencia
    plt.subplot(1, 3, 1)
    plt.title('Imagen de Referencia')
    plt.imshow(img_ref, cmap='gray')
    plt.axis('off')

    # Imagen original a alinear
    plt.subplot(1, 3, 2)
    plt.title('Imagen Original a Alinear')
    plt.imshow(img_mov, cmap='gray')
    plt.axis('off')

    # Imagen alineada
    plt.subplot(1, 3, 3)
    plt.title('Imagen Alineada (Afín)')
    plt.imshow(aligned_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Visualización adicional: Overlay Transparente
    plt.figure(figsize=(6, 6))
    plt.title('Overlay de Imágenes Alineadas')
    plt.imshow(img_ref, cmap='gray', alpha=0.5, label='Referencia')
    plt.imshow(aligned_img, cmap='jet', alpha=0.5, label='Alineada')
    plt.axis('off')
    plt.show()

    # Imagen de diferencia
    difference = img_ref - aligned_img
    plt.figure(figsize=(6, 6))
    plt.title('Imagen de Diferencia')
    plt.imshow(difference, cmap='bwr')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return