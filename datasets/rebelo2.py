from functools import reduce

import os

from datasets.dirdataset import DirDataset


#         min    3^3    2^3
_XSHAPE = 250  # 270  # 256
_YSHAPE = 625  # 648  # 632

_EXCLUDE = ["references", "unknown", "imgs"]
_DEFAULT_DIR = "./downloaded/Rebelo Dataset/database"


def _rebelo_subdirs(image_dir: str) -> list[str]:
    return [os.path.join(image_dir, "real"), os.path.join(image_dir, "syn")]


def RebeloDataset(
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
):
    return DirDataset(
        image_dirs=_rebelo_subdirs(image_dir),
        shape=(_XSHAPE, _YSHAPE),
        exclude=_EXCLUDE,
        multiply_of=multiply_of,
    )


def RebeloDatasetOneCat(
        category: str | int,
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
):
    image_dirs = _rebelo_subdirs(image_dir)
    listdirs = [set(os.listdir(dirr)) for dirr in image_dirs]

    categories = list(sorted(set(reduce(lambda a, b: a.union(b), listdirs)).difference(_EXCLUDE)))

    if isinstance(category, int):
        categories = categories[:category] + categories[category+1:]
    else:
        categories.remove(category)

    return DirDataset(
        image_dirs=[os.path.join(image_dir, "real"), os.path.join(image_dir, "syn")],
        shape=(0, 0),
        exclude=_EXCLUDE + categories,
        multiply_of=multiply_of,
    )

# Bounding boxes
# flat 110 40
# notesFlags 250 57
# rests1 150 56
# altoClef 89 56
# rests2 136 61
# trebleClef 204 83
# naturals 119 38
# notesOpen 147 51
# staccatissimo 20 17
# sharps 108 56
# accent 35 48
# beams 127 208
# time 58 54
# bassClef 79 38
# relation 85 625
# notes 190 55
