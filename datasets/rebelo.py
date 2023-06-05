import numpy as np
import tensorflow as tf
import os

_IMAGE_DIR = "./downloaded/Rebelo Dataset/MainClasses"

categories = []

X = []
y = []

for category in os.listdir(_IMAGE_DIR):

    list_ds = tf.data.Dataset.list_files(str(os.path.join(_IMAGE_DIR, category, '*.png')))
    for f in list_ds:
        image = tf.io.read_file(f)
        image = tf.io.decode_png(image, 1)
        X.append(1 - image/255)
        y.append(len(categories))

    categories.append(category)

N_OF_CATEGORIES = len(categories)
X = tf.stack(X)
y = np.array(y)

shape = list(X[0].shape)

# Accent 458
# AltoCleff 208
# BarLines 524
# BassClef 261
# Beams 508
# Breve 26
# Dots 348
# Flat 413
# Naturals 456
# Notes 451
# NotesFlags 234
# NotesOpen 331
# Relations 528
# Rests1 199
# Rests2 498
# SemiBreve 148
# Sharps 442
# TimeSignatureL 409
# TimeSignatureN 270
# TrebleClef 396
# SUM 7108
