import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def _concat_and_save(images, filename: str = "img.png"):
    columns = []
    for i in images:
        columns.append(np.concatenate(i, axis=0))

    ans = np.concatenate(columns, axis=1)

    plt.imshow(ans, interpolation='nearest')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def gs_img_2d_ls_visualizer(network: tf.keras.Model, n_of_images: int = 10, filename: str = "img.png") -> ():
    """
        Visualise grid of grayscale images generated from 2D latent space.
    """
    images = []
    for i in range(n_of_images):
        images.append([])
        for j in range(n_of_images):
            images[-1].append(network(np.array([i/n_of_images, j/n_of_images])[None])[0])

    _concat_and_save(images, filename)


def gs_img_3d_ls_visualizer(network: tf.keras.Model, n_of_images: int = 10, filename: str = "img.png"):
    """
        Visualise grid of grayscale images generated from 3D latent space.
    """
    for k in range(n_of_images):
        images = []
        for i in range(n_of_images):
            images.append([])
            for j in range(n_of_images):
                images[-1].append(network(np.array([i/n_of_images, j/n_of_images, k/n_of_images])[None])[0])

        _concat_and_save(images, str(k) + filename)


def gs_img_nd_ls_visualizer(
        network: tf.keras.Model, shape: tuple[int, ...], n_of_images: int = 10, filename: str = "img.png"
):
    """
        Visualise grid of grayscale images generated randomly from multidimensional latent space.
    """
    images = []
    for i in range(n_of_images):
        images.append([])
        for j in range(n_of_images):
            images[-1].append(network(np.random.standard_normal(*shape)[None])[0])

    _concat_and_save(images, filename)
