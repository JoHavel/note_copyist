import argparse
import os

import tensorflow as tf
import numpy as np
import cv2

# from generate_images import bounding_box
from center_images import bounding_box


# @tf.function
def one_step(filename: str, stem_filename: str):
    image = tf.io.decode_png(tf.io.read_file(filename), channels=1)[:, :, 0]
    image = cv2.erode(image.numpy(), np.ones([2, 2]))[:, :]
    bb = bounding_box(image, 255)
    im = image[bb[0]:bb[2], bb[1]:bb[3]]
    first_line = im[0, :]
    note_stem_x = tf.shape(first_line)[0] - 1 - tf.argmax(first_line[::-1], output_type=tf.int32)
    tf.io.write_file(
        stem_filename,
        tf.strings.join([
            tf.as_string(bb[1] + note_stem_x),
            tf.as_string(bb[0]),
        ], separator=" ")
    )


def add_other_end_of_stems(directories: list[str]):
    for directory in directories:
        for file in filter(lambda it: it.endswith(".png"), os.listdir(directory)):
            file = os.path.join(directory, file)
            one_step(tf.constant(file), tf.constant(file[:-4]+"-stem_head.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()

    add_other_end_of_stems(args.dirs)
