from enum import StrEnum
from sys import stderr
from typing import Callable

import tensorflow as tf
import os
from argparse import ArgumentParser, Namespace

from datasets.dirdataset import DirDataset
from generators.generator import Generator
from utils.faster_visualizers import gs_img_2d_ls_visualizer, gs_img_3d_ls_visualizer, gs_img_nd_ls_visualizer, \
    cat_gs_img_2d_ls_visualizer, cat_gs_img_nd_ls_visualizer

from datasets.dataset import Dataset, CategoricalDataset
from datasets.mnist import MnistDataset
from datasets import rebelo, rebelo2, centered_rebelo

from generators.basicGAN import GAN
from generators.categoricalGAN import GAN as CGAN
from generators.basicVAE import VAE
from generators.categoricalVAE import VAE as CVAE
from generators.basicAAE import AAE
from generators.categoricalAAE import AAE as CAAE


from parts.decoder import Decoder
from parts.encoder import Encoder2Normal
from parts.discriminator import Discriminator
from parts.cat_discriminator import CatDiscriminator
from utils.my_typing import String

DEFAULT_SEED = 42
""" The default random seed """
DEFAULT_THREADS = 3
""" The default number of threads to use. """
DEFAULT_TF_CPP_MIN_LOG_LEVEL = 2
""" The default TF logging level """


def add_tensorflow_args(parser: ArgumentParser) -> None:
    """ Adds tensorflow settings to `parser` """
    parser.add_argument("--seed", type=int, help="Seed for experiment", default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, help="Number of threads for experiment", default=DEFAULT_THREADS)
    parser.add_argument("--log_level", type=int, help="Log level of TensorFlow", choices=range(0, 4+1), default=DEFAULT_TF_CPP_MIN_LOG_LEVEL)


def set_tensorflow(
        seed: int = DEFAULT_SEED, threads: int = DEFAULT_THREADS, log_level: int = DEFAULT_TF_CPP_MIN_LOG_LEVEL,
        args: Namespace | None = None,
) -> None:
    """ Sets tensorflow settings from `args` or particular settings """
    if args is not None:
        seed = args.seed
        threads = args.threads
        log_level = args.log_level

    # From npfl114
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(log_level))  # Report only TF errors by default
    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)


class CategoryStyle(StrEnum):
    """ Type of processing categories """

    BASIC = "basic"
    """ Without categories """
    CATEGORICAL = "cat"
    """ Categorical """
    BASIC_FOR_EVERY_CAT = "onecat"
    """ Individual model for every category """


_DATASETS = {
    "mnist": MnistDataset,
    "rebelo1": rebelo.RebeloDataset,
    "rebelo2": rebelo2.RebeloDataset,
    "crebelo": centered_rebelo.CenteredRebeloDataset,
    "other": lambda **kwargs: DirDataset(image_dirs="downloaded/other/", string="other", shape=(0, 0), **kwargs),
}
""" Possible datasets for experiments """
_DATASETS_ONECAT = {
    "mnist": lambda category, **kwargs: MnistDataset(**kwargs).one_category(category),
    "rebelo1": lambda category, **kwargs: rebelo.RebeloDataset(**kwargs).one_category(category),
    "rebelo2": lambda category, **kwargs: rebelo2.RebeloDataset(category=category, **kwargs),
    "crebelo": centered_rebelo.CenteredRebeloDataset,
    "other": lambda category, **kwargs: DirDataset(category=category, image_dirs="downloaded/other/", string="other", shape=(0, 0), **kwargs),
}
""" Possible datasets providing particular categories for experiments """

_OUT = "out"
""" Where output directory tree """
_IMAGE_DIR = "images"
""" Deepest directory containing images """
_MODEL_DIR = "parts"
""" Deepest directory containing saved models """


class Experiment:
    """ Class allowing doing experiments with various settings """

    @staticmethod
    def add_network_args(parser: ArgumentParser) -> None:
        """ Add `generators` parameters to `parser` """
        parser.add_argument("--network", type=str, help="Network type", default="vae",
                            choices=["gan", "vae", "aae"])
        parser.add_argument("--latent", default=[2], type=int, nargs="*",
                            help="Latent shape")

        parser.add_argument("--layers", default=[500, 500], type=int, nargs="*",
                            help="Hidden layers of generator (and reversed of dis/enc is second_layers is not specified).")
        parser.add_argument("--conv_layers", default=[], type=int, nargs="*",
                            help="Convolutional layers of generator (and reversed of dis/enc is second_layers is not specified).")
        parser.add_argument("--stride", type=int, help="Stride of generator and dis/enc if second_stride is not specified",
                            default=2)
        parser.add_argument("--kernel", type=int, help="Kernel size of generator and dis/enc if second_kernel is not specified",
                            default=3)
        parser.add_argument("--second_layers", default=None, type=int, nargs="*",
                            help="Hidden layers of discriminator (GAN) or encoder (VAE or AAE).")
        parser.add_argument("--second_conv_layers", default=None, type=int, nargs="*",
                            help="Convolutional layers of discriminator (GAN) or encoder (VAE or AAE).")
        parser.add_argument("--second_stride", type=int, help="Stride of discriminator (GAN) or encoder (VAE or AAE).",
                            default=None)
        parser.add_argument("--second_kernel", type=int, help="Kernel size of discriminator (GAN) or encoder (VAE or AAE).",
                            default=None)
        parser.add_argument("--dis_layers", default=[128], type=int, nargs="*",
                            help="Hidden layers of discriminator (AAE).")
        parser.add_argument("--dis_conv_layers", default=[], type=int, nargs="*",
                            help="Convolutional layers of discriminator (AAE).")
        parser.add_argument("--dis_stride", type=int, help="Stride of discriminator (AAE).", default=2)
        parser.add_argument("--dis_kernel", type=int, help="Kernel size of discriminator (AAE).", default=3)

    @staticmethod
    def add_experiment_args(parser: ArgumentParser) -> None:
        """ Add experiment parameters to `parser` """
        add_tensorflow_args(parser)
        parser.add_argument("--cat", type=CategoryStyle, help="What type of experiment use", default=CategoryStyle.CATEGORICAL,
                            choices=CategoryStyle.__members__.values())
        parser.add_argument("--dataset", type=str, help="Dataset for experiment", default="mnist",
                            choices=_DATASETS.keys() | _DATASETS_ONECAT.keys())
        parser.add_argument("--batch", type=int, help="Batch size", default=50)
        parser.add_argument("--epochs", type=int, help="Number of training epochs", default=50)
        parser.add_argument("--multiply_of", type=int, help="If shape of image may be multiply of something (magnification of conv layers)", default=None)
        parser.add_argument("--n_of_images", type=int, help="Number of images in image grid in example images", default=10)
        parser.add_argument("--directory", type=str, help="Directory where save images and model", default=None)
        parser.add_argument("--category", type=int, help="If cat == basic, here we can set specific category", default=None)
        Experiment.add_network_args(parser)

    cat: CategoryStyle
    batch_size: int
    epochs: int
    directory: str
    visualizer: Callable[[str], None]
    category: int | None
    n_of_images: int
    category: int | None
    latent_shape: tuple[int, ...]
    dataset: Dataset | None
    dataset_generator: Callable[[int], Dataset] | None
    network: tf.keras.Model | String | None
    network_generator: Callable[[Dataset], Decoder] | None
    multiply_of: int | None

    def __init__(
            self,
            cat: CategoryStyle | None = None,
            dataset: Dataset | None = None,
            dataset_generator: Callable[[int], Dataset] | None = None,
            network: tf.keras.Model | String | None = None,
            network_generator: Callable[[Dataset], Decoder] | None = None,
            batch_size: int | None = None,
            epochs: int | None = None,
            directory: str | None = None,
            visualizer: Callable[[str, tf.keras.Model], None] | None = None,
            category: int | None = None,
            n_of_images: int = None,
            latent_shape: tuple[int, ...] | None = None,
            multiply_of: int | None = None,
            args: Namespace | None = None,
    ):
        # TODO check args
        if args is None:
            parser = ArgumentParser()
            self.add_experiment_args(parser)
            args = parser.parse_args()
            set_tensorflow(args=args)

        if cat is None:
            cat = args.cat
        self.cat = cat

        if batch_size is None:
            batch_size = args.batch
        self.batch_size = batch_size

        if epochs is None:
            epochs = args.epochs
        self.epochs = epochs

        if multiply_of is None:
            multiply_of = args.multiply_of
        self.multiply_of = multiply_of

        if cat != CategoryStyle.BASIC_FOR_EVERY_CAT:
            if category is None:
                category = args.category
            self.category = category
        else:
            self.category = 0

        if dataset is None and dataset_generator is None:
            if self.category is not None:
                dataset_generator = _DATASETS_ONECAT[args.dataset]
            else:
                if multiply_of is None:
                    dataset = _DATASETS[args.dataset]()
                else:
                    dataset = _DATASETS[args.dataset](multiply_of=multiply_of)
        self.dataset_generator = dataset_generator

        if dataset is None:
            if self.multiply_of is None:
                dataset = dataset_generator(self.category)
            else:
                dataset = dataset_generator(self.category, multiply_of=multiply_of)
        self.dataset = dataset

        if latent_shape is None:
            if network is not None:
                latent_shape = network.latent_shape
            else:
                latent_shape = args.latent
        self.latent_shape = latent_shape

        if network is None and network_generator is None:
            network_generator = self.get_network_generator(args)
        self.network_generator = network_generator

        if network is None:
            network = network_generator(dataset)
        self.network = network

        if args is not None and directory is None:
            directory = args.directory
        self.directory = self.get_and_create_directory(directory)

        if visualizer is None:
            if n_of_images is None:
                n_of_images = args.n_of_images
            visualizer = self.get_visualizer(n_of_images)
        self.visualizer = visualizer

    def get_network_generator(self, args: Namespace) -> Callable[[Dataset], Generator]:
        """ Returns creator of chosen `generators` """
        hidden_layers = args.layers
        conv_layers = args.conv_layers
        stride = args.stride
        kernel_size = args.kernel
        second_hidden_layers = args.second_layers if args.second_layers is not None else list(reversed(hidden_layers))
        second_conv_layers = args.second_conv_layers if args.second_conv_layers is not None else list(reversed(conv_layers))
        second_stride = args.second_stride if args.second_stride is not None else stride
        second_kernel_size = args.second_kernel if args.second_kernel is not None else kernel_size

        match args.network:
            case "gan":
                if self.cat == CategoryStyle.CATEGORICAL:
                    def network_generator(dataset: CategoricalDataset) -> Decoder:
                        return CGAN(
                            Decoder(
                                (self.latent_shape[0] + dataset.n_of_categories,), dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                            CatDiscriminator(
                                (dataset.shape, dataset.n_of_categories),
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                            dataset.n_of_categories,
                        )
                else:
                    def network_generator(dataset: Dataset) -> Decoder:
                        return GAN(
                            Decoder(
                                self.latent_shape, dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                            Discriminator(
                                dataset.shape,
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                        )
            case "vae":
                if self.cat == CategoryStyle.CATEGORICAL:
                    def network_generator(dataset: CategoricalDataset) -> Decoder:
                        network = CVAE(
                            Encoder2Normal(
                                dataset.shape, (self.latent_shape[0] + dataset.n_of_categories,),
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                            Decoder(
                                (self.latent_shape[0] + dataset.n_of_categories,), dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                            dataset.n_of_categories,
                        )
                        network.compile(optimizer=tf.optimizers.Adam())
                        return network
                else:
                    def network_generator(dataset: Dataset) -> Decoder:
                        network = VAE(
                            Encoder2Normal(
                                dataset.shape, self.latent_shape,
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                            Decoder(
                                self.latent_shape, dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                        )
                        network.compile(optimizer=tf.optimizers.Adam())
                        return network
            case "aae":
                if self.cat == CategoryStyle.CATEGORICAL:
                    def network_generator(dataset: CategoricalDataset) -> Decoder:
                        network = CAAE(
                            Encoder2Normal(
                                dataset.shape, (self.latent_shape[0] + dataset.n_of_categories,),
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                            Decoder(
                                (self.latent_shape[0] + dataset.n_of_categories,), dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                            Discriminator(
                                self.latent_shape,
                                hidden_layers=args.dis_layers, conv_layers=args.dis_conv_layers,
                                strides=args.dis_stride, kernel_sizes=args.dis_kernel,
                            ),
                            dataset.n_of_categories,
                        )
                        network.compile(optimizer=tf.optimizers.Adam())
                        return network
                else:
                    def network_generator(dataset: Dataset) -> Decoder:
                        network = AAE(
                            Encoder2Normal(
                                dataset.shape, self.latent_shape,
                                hidden_layers=second_hidden_layers, conv_layers=second_conv_layers,
                                strides=second_stride, kernel_sizes=second_kernel_size,
                            ),
                            Decoder(
                                self.latent_shape, dataset.shape,
                                hidden_layers=hidden_layers, conv_layers=conv_layers,
                                strides=stride, kernel_sizes=kernel_size,
                            ),
                            Discriminator(
                                self.latent_shape,
                                hidden_layers=args.dis_layers, conv_layers=args.dis_conv_layers,
                                strides=args.dis_stride, kernel_sizes=args.dis_kernel,
                            ),
                        )
                        network.compile(optimizer=tf.optimizers.Adam())
                        return network
            case _:
                raise ValueError("Invalid network name!")

        return network_generator

    def get_visualizer(self, n_of_images: int) -> Callable[[str, tf.keras.Model], None]:
        """ Gets suitable visualizer """
        if self.cat == CategoryStyle.CATEGORICAL:
            if sum(self.latent_shape) == 2:
                @tf.function
                def visualizer(filename, network) -> None:
                    cat_gs_img_2d_ls_visualizer(
                        network, network.n_of_categories, n_of_images, filename
                    )
            else:
                @tf.function
                def visualizer(filename, network) -> None:
                    cat_gs_img_nd_ls_visualizer(
                        network, network.n_of_categories, (self.latent_shape[0] + self.dataset.n_of_categories,), n_of_images, filename
                    )
        else:
            if sum(self.latent_shape) == 2:
                @tf.function
                def visualizer(filename, network) -> None:
                    gs_img_2d_ls_visualizer(network, n_of_images, filename + ".png")
            else:
                if sum(self.latent_shape) == 3:
                    @tf.function
                    def visualizer(filename, network) -> None:
                        gs_img_3d_ls_visualizer(network, n_of_images, filename + ".png")
                else:
                    @tf.function
                    def visualizer(filename, network) -> None:
                        gs_img_nd_ls_visualizer(network, self.latent_shape, n_of_images, filename + ".png")
        return visualizer

    def get_and_create_directory(self, directory: str | None) -> str:
        """ Gets directory name from settings (if `directory` is None) and creates it """
        if directory is None:
            directory = os.path.join(_OUT, self.dataset.string, str(self.cat), self.network.string + "b" + str(self.batch_size))

        if os.path.exists(directory):
            if input(f"Directory `{directory}` already exists, do you want to continue? y/n") != "y":
                raise ValueError(f"Directory `{directory}` already exists!")

        os.makedirs(os.path.join(directory, _IMAGE_DIR), exist_ok=True)
        print(f"Using directory `{directory}`.", file=stderr)
        return directory

    def run(self) -> None:
        """ Runs experiment """
        def draw(i, _) -> None:
            self.network.save_all(
                os.path.join(self.directory, _MODEL_DIR, f"e{i+1}{'' if self.category is None else f'c{self.category}'}")
            )
            self.visualizer(
                tf.constant(os.path.join(self.directory, _IMAGE_DIR, f"e{i+1}{'' if self.category is None else f'c{self.category}'}")),
                self.network,
            )

        train_len = len(self.dataset.X_train)
        if train_len > self.batch_size:
            train_len -= train_len % self.batch_size

        if self.cat == CategoryStyle.CATEGORICAL:
            self.network.fit(
                self.dataset.X_train[:train_len],
                self.dataset.y_train[:train_len],
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
            )
        else:
            self.network.fit(
                self.dataset.X_train[:train_len],
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=tf.keras.callbacks.LambdaCallback(on_epoch_end=draw)
            )

        if self.cat == CategoryStyle.BASIC_FOR_EVERY_CAT:
            self.category += 1
            if self.multiply_of is None:
                self.dataset = self.dataset_generator(self.category)
            else:
                self.dataset = self.dataset_generator(self.category, multiply_of=self.multiply_of)

            if len(self.dataset.X_train) == 0:
                return
            self.network = self.network_generator(self.dataset)
            self.run()
