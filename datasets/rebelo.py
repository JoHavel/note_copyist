import numpy as np
import tensorflow as tf
import os

from datasets.dataset import CategoricalDataset, DatasetPart


class RebeloDataset(CategoricalDataset):
    def __init__(self, image_dir: str = "./downloaded/Rebelo Dataset/MainClasses"):
        self.categories = []

        X = []
        y = []

        for category in os.listdir(image_dir):

            list_ds = tf.data.Dataset.list_files(str(os.path.join(image_dir, category, '*.png')))
            for f in list_ds:
                image = tf.io.read_file(f)
                image = tf.io.decode_png(image, 1)
                X.append(1 - image/255)
                y.append(len(self.categories))

            self.categories.append(category)

        X = tf.stack(X)
        y = np.array(y)

        shape = X[0].shape

        X = {DatasetPart.TRAIN: X}
        y = {DatasetPart.TRAIN: tf.constant(y)}
        super().__init__(shape, X, y, len(self.categories))

# Counts
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
