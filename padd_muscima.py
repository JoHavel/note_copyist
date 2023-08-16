import argparse
import os
import random
import shutil

import cv2
import tensorflow as tf

import datasets.centered_rebelo
from center_images import center_image_to_point
from experiments.experiment import set_tensorflow, add_tensorflow_args
from generate_images_for_mashcima import N_OF_SYMBOLS_IN_MUSCIMA_PP


def rebelo(name: str, output_dir: str, number: int, offset: int) -> None:
    input_dir = os.path.join(datasets.centered_rebelo._DEFAULT_DIR, name)
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("png")]
    files.sort()
    random.shuffle(files)

    rebelo_number = len(files)

    for i in range(number):
        image_file = os.path.join(output_dir, f"im{i + offset}.png")
        shutil.copy(files[i % rebelo_number], image_file)


def muscima_pp(name: str, output_dir: str, number: int, offset: int) -> None:
    input_dir = os.path.join("downloaded", "muscima-exported-symbols", name)
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("png")]
    files.sort()
    random.shuffle(files)

    for i in range(number):
        image_file = os.path.join(output_dir, f"im{i + offset}.png")
        image = 255 - cv2.imread(files[i], flags=cv2.IMREAD_GRAYSCALE)

        with open(files[i][:-4] + ".txt") as file:
            center_x, center_y = map(int, file.readline().split())

        image = center_image_to_point(image, center_x, center_y, pad_value=255)
        cv2.imwrite(os.path.join(image_file), image)


def padd_muscima(args):
    for i, name in enumerate(sorted(list(N_OF_SYMBOLS_IN_MUSCIMA_PP))):
        number = N_OF_SYMBOLS_IN_MUSCIMA_PP[name]

        _output_dir = os.path.join(args.output_dir, name)
        os.makedirs(_output_dir, exist_ok=True)

        first = True
        offset = 0
        for source in args.source:
            _number = number // len(args.source)
            if first:
                first = False
                _number += number % len(args.source)

            set_tensorflow(args=args)
            if source == "REBELO":
                rebelo(name, _output_dir, _number, offset)
            elif source == "MUSCIMA":
                muscima_pp(name, _output_dir, _number, offset)
            else:
                raise Exception()

            offset += _number


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("source", type=str, help="REBELO or MUSCIMA", nargs="+", choices=["REBELO", "MUSCIMA"])
    add_tensorflow_args(parser)
    args = parser.parse_args()

    padd_muscima(args)
