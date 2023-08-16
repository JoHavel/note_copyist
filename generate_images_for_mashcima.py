import argparse
import os
import random
import shutil

import tensorflow as tf

import datasets.centered_rebelo
from experiments.experiment import set_tensorflow, add_tensorflow_args
from generate_images import generate_images, bounding_box
from add_other_end_of_stem import add_other_end_of_stems

N_OF_SYMBOLS_IN_MUSCIMA_PP: dict[str, int] = {
    "sharp": 1_689,
    "flat": 1_064,
    "natural": 1_021,
    "g-clef": 341,
    "f-clef": 250,
    "c-clef": 155,
    "half-note": 845,
    "quarter-note": 15_424,
    "eighth-note-up": 1_697,
    "eighth-note-down": 1_697,
    "quarter-rest": 553,
    "whole-note": 1_183,
}
""" Dictionary: symbol name -> the number of symbols in MUSCIMA++ ignoring the evaluation writers, see https://github.com/johavel/BachelorThesis """


@tf.function
def rebelo_one_step(input_filename: str, img_filename: str, center_filename: str) -> None:
    out = tf.io.decode_png(tf.io.read_file(input_filename), channels=1)
    bb = bounding_box(out, half=255//2)
    im = out[bb[0]:bb[2], bb[1]:bb[3]]
    image = tf.image.encode_png(tf.cast(im, tf.uint8))
    tf.io.write_file(img_filename, image)
    tf.io.write_file(
        center_filename,
        tf.strings.join([
            tf.as_string(tf.shape(out, out_type=tf.int64)[1]//2 - bb[1]),
            tf.as_string(tf.shape(out, out_type=tf.int64)[0]//2 - bb[0]),
        ], separator=" ")
    )


def rebelo(name: str, output_dir: str, number: int, offset: int) -> None:
    input_dir = os.path.join(datasets.centered_rebelo._DEFAULT_DIR, name)
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir)]
    files.sort()
    random.shuffle(files)

    rebelo_number = len(files)
    images = []
    centers = []

    for i in range(min(rebelo_number, number)):
        image_file = os.path.join(output_dir, f"im{i + offset}.png")
        center_file = os.path.join(output_dir, f"im{i + offset}.txt")
        rebelo_one_step(
            tf.constant(files[i]),
            tf.constant(image_file),
            tf.constant(center_file),
        )
        images.append(image_file)
        centers.append(center_file)

    number -= rebelo_number
    offset += rebelo_number

    for i in range(number):
        image_file = os.path.join(output_dir, f"im{i + offset}.png")
        center_file = os.path.join(output_dir, f"im{i + offset}.txt")
        shutil.copy(images[i % rebelo_number], image_file)
        shutil.copy(centers[i % rebelo_number], center_file)


def muscima_pp(name: str, output_dir: str, number: int, offset: int) -> None:
    input_dir = os.path.join("downloaded", "muscima-exported-symbols", name)
    files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith("png")]
    files.sort()
    random.shuffle(files)

    stem = name in {"half-note", "quarter-note", "eighth-note-up", "eighth-note-down"}

    for i in range(number):
        image_file = os.path.join(output_dir, f"im{i + offset}.png")
        center_file = os.path.join(output_dir, f"im{i + offset}.txt")
        shutil.copy(files[i], image_file)  # Maybe % len(files)
        shutil.copy(files[i][:-4] + ".txt", center_file)  # Maybe % len(files)
        if stem:
            stem_file = os.path.join(output_dir, f"im{i + offset}-stem_head.txt")
            shutil.copy(files[i][:-4] + "-stem_head.txt", stem_file)  # Maybe % len(files)


def generate(args, sources, networks, cats) -> None:
    for i, name in enumerate(sorted(list(N_OF_SYMBOLS_IN_MUSCIMA_PP))):
        number = N_OF_SYMBOLS_IN_MUSCIMA_PP[name]

        _output_dir = os.path.join(args.output_dir, name)
        os.makedirs(_output_dir, exist_ok=True)

        first = True
        offset = 0
        for source, network, cat in zip(sources, networks, cats):
            _number = number // len(sources)
            if first:
                first = False
                _number += number % len(sources)

            set_tensorflow(args=args)
            if source == "REBELO":
                rebelo(name, _output_dir, _number, offset)
            elif source == "MUSCIMA":
                muscima_pp(name, _output_dir, _number, offset)
            else:
                assert cat != "_" and network != "_"
                if cat == "onecat":
                    generate_images(source+"c"+str(i), _output_dir, _number, network)
                else:
                    category_one_hot = tf.one_hot(i, len(N_OF_SYMBOLS_IN_MUSCIMA_PP))
                    generate_images(source, _output_dir, _number, network, category_one_hot)

            offset += _number

        if name in {"half-note", "quarter-note", "eighth-note-up", "eighth-note-down"}:
            add_other_end_of_stems([_output_dir])

        print(f"Generating `{name}` done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("source", type=str, help="network directory or REBELO or MUSCIMA", nargs="+")
    parser.add_argument("--network", type=str, help="Network type", nargs="*", choices=["gan", "vae", "aae", "_"], default=[])
    parser.add_argument("--cat", type=str, help="Cat or onecat?", nargs="*", choices=["cat", "onecat", "_"], default=[])
    add_tensorflow_args(parser)
    args = parser.parse_args()

    sources = args.source
    networks = args.network
    assert len(networks) <= len(sources)
    networks += ["_"]*(len(sources) - len(networks))
    cats = args.cat
    assert len(cats) <= len(sources)
    cats += ["_"]*(len(sources) - len(cats))

    generate(args, sources, networks, cats)

