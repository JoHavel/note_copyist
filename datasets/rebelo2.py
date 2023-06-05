from functools import reduce

import numpy as np
import tensorflow as tf
import os

_IMAGE_DIR = [
    "./downloaded/Rebelo Dataset/database/real",
    "./downloaded/Rebelo Dataset/database/syn",
]

#        3^3   2^3   minimum
XSHAPE = 270  # 256  # 250
YSHAPE = 648  # 632  # 625

categories = []

X = []
y = []

listdirs = [set(os.listdir(dirr)) for dirr in _IMAGE_DIR]

for category in set(reduce(lambda a, b: a.union(b), listdirs)):
    if category == "references" or category == "unknown" or category == "imgs":
        continue

    list_ds = [filename for i in range(len(_IMAGE_DIR)) if category in listdirs[i] for filename in tf.data.Dataset.list_files(os.path.join(_IMAGE_DIR[i], category, '*.png'))]
    for f in list_ds:
        image = tf.io.read_file(f)
        image = tf.io.decode_png(image, 1)
        image = 1 - image/255
        image_x_shape = image.shape[0]
        image_y_shape = image.shape[1]
        image = tf.image.pad_to_bounding_box(image, (XSHAPE - image_x_shape)//2, (YSHAPE - image_y_shape)//2, XSHAPE, YSHAPE)
        X.append(image)
        y.append(len(categories))

    categories.append(category)

N_OF_CATEGORIES = len(categories)
X = tf.stack(X)
y = np.array(y)

shape = list(X[0].shape)
