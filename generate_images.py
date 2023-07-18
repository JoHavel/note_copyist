import argparse
import os
import tensorflow as tf

from datasets.centered_rebelo import FUNCTIONS
from experiments.experiment import set_tensorflow, add_tensorflow_args
import generators.generator
from generators import basicGAN, basicVAE, basicAAE, categoricalGAN, categoricalVAE, categoricalAAE
from utils.binarize import Binarize

NETWORKS: dict[str, generators.generator.Generator] =\
    {"gan": basicGAN.GAN, "vae": basicVAE.VAE, "aae": basicAAE.AAE}

CNETWORKS: dict[str, generators.generator.Generator] =\
    {"gan": categoricalGAN.GAN, "vae": categoricalVAE.VAE, "aae": categoricalAAE.AAE}


@tf.function
def bounding_box(image) -> tuple[int, int, int, int]:
    binary_image = tf.cast(image[..., 0] > 0.5, tf.int8)
    top = tf.reduce_min(tf.argmax(tf.pad(binary_image, [[0, 1], [0, 0]], constant_values=1)))
    left = tf.reduce_min(tf.argmax(tf.pad(binary_image, [[0, 0], [0, 1]], constant_values=1), axis=1))
    binary_image = tf.reverse(binary_image, (0, 1))
    bottom = image.shape[0] - tf.reduce_min(tf.argmax(tf.pad(binary_image, [[0, 1], [0, 0]], constant_values=1)))
    right = image.shape[1] - tf.reduce_min(tf.argmax(tf.pad(binary_image, [[0, 0], [0, 1]], constant_values=1), axis=1))
    return top, left, bottom, right


@tf.function
def one_step(model, img_filename: str, center_filename, category: tf.Tensor):
    inp = tf.concat((category, model.model._latent_prior.sample()), -1)[None, ...]
    out = model(inp)[0, ..., None]
    bb = bounding_box(out)
    im = 255 * out[bb[0]:bb[2], bb[1]:bb[3]]
    image = tf.image.encode_png(tf.cast(im, tf.uint8))
    tf.io.write_file(img_filename, image)
    tf.io.write_file(
        center_filename,
        tf.strings.join([
            tf.as_string(out.shape[1]//2 - bb[1]),
            tf.as_string(out.shape[0]//2 - bb[0]),
        ], separator=" ")
    )


def generate_images(
        input_file: str, output_dir: str, number: int, network: str,
        category: tf.Tensor = None
):
    os.makedirs(output_dir, exist_ok=True)
    if category is None:
        category = tf.constant(())
        model = Binarize(NETWORKS[network].load_all(input_file), 0.2)
    else:
        model = Binarize(CNETWORKS[network].load_all(input_file), 0.2)

    for i in range(number):
        one_step(
            model,
            tf.constant(os.path.join(output_dir, f"im{i}.png")),
            tf.constant(os.path.join(output_dir, f"im{i}.txt")),
            category
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("number", type=int)
    parser.add_argument("--network", type=str, help="Network type", default="vae", choices=["gan", "vae", "aae"])
    parser.add_argument("--cat", type=str, help="Cat or onecat?", default="onecat", choices=["cat", "onecat"])
    add_tensorflow_args(parser)
    args = parser.parse_args()

    for i, name in enumerate(sorted(list(FUNCTIONS)) + ["whole-note"]):
        set_tensorflow(args=args)
        if args.cat == "onecat":
            generate_images(args.input_file+"c"+str(i), os.path.join(args.output_dir, name), args.number, args.network)
        else:
            category_one_hot = tf.one_hot(i, len(FUNCTIONS) + 1)
            generate_images(args.input_file, os.path.join(args.output_dir, name), args.number, args.network, category_one_hot)



