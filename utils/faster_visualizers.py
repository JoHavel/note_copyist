# Copy of visualizers.py, optimized for tf (arguments must be tf.constants)

import tensorflow as tf
import numpy as np


@tf.function
def _concat_and_save(images, filename: str = "img.png"):
    """ Concat grid of images to one image and show/save it.
        image is saved to filename (directory must exist!)
    """
    columns = []
    for i in images:
        columns.append(tf.concat(i, axis=0))

    ans = tf.concat(columns, axis=1)

    if len(ans.shape) == 2:
        ans = ans[..., None]

    image = tf.image.encode_png(tf.cast(255*ans, tf.uint8))
    tf.io.write_file(filename, image)


@tf.function
def gs_img_2d_ls_visualizer(network: tf.keras.Model, n_of_images: int = 10, filename: str = "img.png") -> ():
    """
        Visualise grid of grayscale images generated from 2D latent space.
    """
    images = []
    for i in range(n_of_images):
        images.append([])
        for j in range(n_of_images):
            images[-1].append(network(np.array([2*(i/(n_of_images - 1) - 0.5), 2*(j/(n_of_images - 1) - 0.5)])[None])[0])

    _concat_and_save(images, filename)


@tf.function
def gs_img_3d_ls_visualizer(network: tf.keras.Model, n_of_images: int = 10, filename: str = "img", extension: str = ".png"):
    """
        Visualise grid of grayscale images generated from 3D latent space.
    """
    for k in range(n_of_images):
        images = []
        for i in range(n_of_images):
            images.append([])
            for j in range(n_of_images):
                images[-1].append(network(np.array([i/n_of_images, j/n_of_images, k/n_of_images])[None])[0])

        _concat_and_save(images, (filename + "d" + str(k) + extension))


@tf.function
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


@tf.function
def cat_gs_img_nd_ls_visualizer(
        network: tf.keras.Model,
        n_of_categories: int,
        shape: tuple[int, ...],
        n_of_images: int = 10,
        filename: str = "img",
        extension: str = ".png",
):
    """
        Visualise grid of grayscale images generated randomly from multidimensional latent space.

        With one_hot categories in first dimensions (FIXME other than shapes [n])
    """
    for c in range(n_of_categories):
        category = np.zeros((n_of_images**2, n_of_categories))
        category[:, c] = 1

        images = network(np.concatenate([
            category,
            np.random.standard_normal((n_of_images**2, shape[0] - n_of_categories))
        ], axis=1))

        images = tf.reshape(images, (n_of_images, n_of_images,) + images.shape[1:])

        _concat_and_save(images, (filename + "c" + str(c) + extension))


@tf.function
def cat_gs_img_2d_ls_visualizer(
        network: tf.keras.Model,
        n_of_categories: int,
        n_of_images: int = 10,
        filename: str = "img",
        extension: str = ".png",
):
    """
        Visualise grid of grayscale images generated randomly from multidimensional latent space.

        With one_hot categories in first dimensions (FIXME other than shapes [n])
    """
    for c in range(n_of_categories):
        images = []
        category = np.zeros(n_of_categories)
        category[c] = 1
        for i in range(n_of_images):
            images.append([])
            for j in range(n_of_images):
                images[-1].append(network(np.concatenate([
                    category,
                    [2*(i/(n_of_images - 1) - 0.5), 2*(j/(n_of_images - 1) - 0.5)]
                ])[None])[0])

        _concat_and_save(images, (filename + "c" + str(c) + extension))
