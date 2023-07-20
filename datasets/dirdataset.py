import os
from functools import reduce

import imagesize

import tensorflow as tf

from datasets.dataset import CategoricalDataset, DatasetPart


class DirDataset(CategoricalDataset):
    """ Dataset from directory (every subdirectory is category) """
    def __init__(
            self,
            image_dirs: list[str] | str,
            string: str,
            shape: tuple[int, int] | tuple[int, int, int] | None = None,
            exclude: list[str] | tuple[str] | str = (),
            inverse: bool = True,
            multiply_of: int | None = None,
            create=lambda *args: (),
            category: str | int | None = None,
    ):
        """
            Loads dataset from `image_dirs` except for `excluded`. It has `String` value `string`.
            All data has `shape` if `shape` is `tuple` and is not `[0, 0]` (automatically finds surrounding shape to
            multiplies of `multiply_of`), None means original shape. Images are `inverse`(d).

            We can load only particular `category` or provide `create` function for creating dataset
            if `image_dirs` are empty.
        """
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]

        if not all([os.path.exists(dirr) for dirr in image_dirs]):
            create()

        if isinstance(exclude, str):
            exclude = [exclude]

        if category is not None:
            listdirs = [set(os.listdir(dirr)) for dirr in image_dirs]
            other_categories = list(sorted(set(reduce(lambda a, b: a.union(b), listdirs)).difference(set(exclude))))

            if isinstance(category, int):
                other_categories = other_categories[:category] + other_categories[category+1:]
            else:
                other_categories.remove(category)

            exclude = other_categories + list(exclude)


        if shape is not None and (shape[0] == 0 and shape[1] == 0):
            shape = (*DirDataset._find_bounding_box(image_dirs, exclude), *shape[2:])

        if multiply_of is not None:
            if shape is None:
                raise ValueError(
                    "If multiply_of is not None, then shape cannot be None (= don't change shape), use (0, 0) instead."
                )

            shape = (
                shape[0] + multiply_of - (shape[0] % multiply_of),
                shape[1] + multiply_of - (shape[1] % multiply_of),
                *shape[2:]
            )

        X, y, self.categories = DirDataset._load_dataset(image_dirs, shape, exclude, inverse)

        if shape is None:
            shape = X[0].shape

        X = {DatasetPart.TRAIN: X}
        y = {DatasetPart.TRAIN: y}
        super().__init__(shape, X, y, string, len(self.categories))

    @staticmethod
    def _load_dataset(
            image_dirs: list[str],
            shape: tuple[int, int] | tuple[int, int, int] | None = None,
            exclude: list[str] | tuple[str] = (),
            inverse: bool = True,
    ) -> (tf.Tensor, tf.Tensor, [str]):
        """ Loads dataset from `image_dirs` except for `exclude`. Images have `shape` and are `inverse`(d). """

        X: list[tf.Tensor] = []
        y: list[int] = []
        categories: list[str] = []

        def process_image_file(f):
            X.append(DirDataset._load_image(f, shape, inverse))
            y.append(len(categories))

        def add_category(category):
            categories.append(category)

        DirDataset._list_categories(image_dirs, exclude, process_image_file, add_category)

        X = tf.stack(X)
        y = tf.constant(y)
        return X, y, categories

    @staticmethod
    def _load_image(
            f,
            shape: tuple[int, int] | tuple[int, int, int] | None = None,
            inverse: bool = True
    ) -> tf.Tensor:
        """ Loads image from file `f`, pads it to `shape` and `inverse` black and white. """
        image = tf.io.read_file(f)
        image = tf.io.decode_png(image, 1 if shape is None or len(shape) == 2 else shape[2])
        image = tf.cast(image, tf.float32)/255

        if inverse:
            image = 1 - image

        if shape is not None:
            image_x_shape = image.shape[0]
            image_y_shape = image.shape[1]
            image = tf.image.pad_to_bounding_box(
                image, (shape[0] - image_x_shape)//2, (shape[1] - image_y_shape)//2, shape[0], shape[1]
            )

        if shape is not None and len(shape) == 2:
            image = image[..., 0]

        return image

    @staticmethod
    def _list_categories(
            image_dirs: list[str],
            exclude: list[str] | tuple[str] = (),
            every_file=lambda *args: None,
            every_category=lambda *args: None,
    ) -> None:
        """
            Does `every_cathegory` on every directory and `every_file` on every file in it.
            Lists all `image_dirs` except for `exclude`.
        """
        listdirs = [set(os.listdir(dirr)) for dirr in image_dirs]

        for category in sorted(set(reduce(lambda a, b: a.union(b), listdirs))):
            if category in exclude:
                continue

            file_list = [
                filename
                for i in range(len(image_dirs))
                if category in listdirs[i]
                for filename in tf.data.Dataset.list_files(os.path.join(image_dirs[i], category, '*.png'), shuffle=False)
            ]

            for f in file_list:
                every_file(f)

            every_category(category)

    @staticmethod
    def _find_bounding_box(
            image_dirs: list[str],
            exclude: list[str] | tuple[str] = (),
    ) -> (int, int):
        """ Finds rectangle containing all images in `image_dirs` except for `exclude` dirs. """
        x = 0
        y = 0

        def process_image_file(f):
            nonlocal x, y
            width, height = imagesize.get(f.numpy())
            x = max(x, height)
            y = max(y, width)

        DirDataset._list_categories(image_dirs, exclude, process_image_file)

        return x, y
