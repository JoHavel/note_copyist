from functools import reduce

import os

from datasets.dirdataset import DirDataset
from datasets.rebelo import _DATASET, download_rebelo

#         min    3^3    2^3
_YSHAPE = 250  # 270  # 256
""" Default height containing all Rebelo images """
_XSHAPE = 625  # 648  # 632
""" Default width containing all Rebelo images """

_EXCLUDE = ["references", "unknown", "imgs"]
""" Excluded directories (it does not contain images of symbols / do not have label) """
_DEFAULT_DIR = os.path.join(_DATASET, "database")
""" Directory for the second Rebelo dataset data """

_STRING = "rebelo2"


def _rebelo_subdirs(image_dir: str) -> list[str]:
    """ From the second Rebelo dataset `image_dir` gets the desired directories """
    return [os.path.join(image_dir, "real"), os.path.join(image_dir, "syn")]


def RebeloDataset(
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
        category: str | int = None,
):
    """
        ``Class'' encapsulating the second (original-sized) Rebelo dataset.
        It loads it from `image_dir` with image sizes divisible by `multiply_of`.
        Loads only `category` (None |-> all).
    """
    return DirDataset(
        image_dirs=_rebelo_subdirs(image_dir),
        string=_STRING,
        shape=(_YSHAPE, _XSHAPE) if category is None else (0, 0),
        exclude=_EXCLUDE,
        multiply_of=multiply_of,
        create=download_rebelo,
        category=category,
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
