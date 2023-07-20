import os

from datasets.dataset import _DOWNLOADED
from datasets.dirdataset import DirDataset

_SHAPE = (20, 20)
""" The original shape of Rebelo dataset images """

_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/Rebelo.Dataset.zip"
""" The URL from where download the Rebelo dataset """
_ZIP_FILE = os.path.join(_DOWNLOADED, "rebelo.zip")
""" Where store downloaded .zip file of the Rebelo dataset."""
_DATASET = os.path.join(_DOWNLOADED, "Rebelo Dataset")
""" Directory for the Rebelo dataset data """
_7ZIP_FILES = [os.path.join(_DATASET, "database1.7z"), os.path.join(_DATASET, "database2.7z"), os.path.join(_DATASET, "database3.7z")]
""" Where store unzipped (from .zip) files """
_DEFAULT_DIR = os.path.join(_DATASET, "MainClasses")
""" Default directory for the first Rebelo dataset """

_STRING = "rebelo1"
""" String for RebeloDataset (for naming directories for models) """


def download_rebelo():
    """ Downloads the Rebelo dataset """
    from sys import stderr
    from requests import get
    from zipfile import ZipFile
    from py7zr import SevenZipFile

    if not os.path.exists(_DOWNLOADED):
        os.mkdir(_DOWNLOADED)

    print("Downloading Rebelo dataset.", file=stderr)
    print("https://github.com/apacha/OMR-Datasets#rebelo-dataset", file=stderr)
    print('A. Rebelo, G. Capela, and J. S. Cardoso, "Optical recognition of music symbols: A comparative study" in International Journal on Document Analysis and Recognition, vol. 13, no. 1, pp. 19-31, 2010. DOI: 10.1007/s10032-009-0100-1 (http://dx.doi.org/10.1007/s10032-009-0100-1)', file=stderr)

    with get(_URL) as zip_request:
        with open(os.path.join(_ZIP_FILE), "wb") as zip_file:
            zip_file.write(zip_request.content)

    print("Extracting Rebelo dataset.", file=stderr)

    with ZipFile(_ZIP_FILE, "r") as zip_file:
        zip_file.extractall(_DOWNLOADED)

    for _7ZIP_FILE in _7ZIP_FILES:
        with SevenZipFile(_7ZIP_FILE, "r") as seven_zip_file:
            seven_zip_file.extractall(_DATASET)

    print("Done: Downloading and extracting Rebelo dataset.", file=stderr)


def RebeloDataset(
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
        category: str | int = None,
):
    """
        ``Class'' encapsulating the first (squared) Rebelo dataset.
        It loads it from `image_dir` with image sizes divisible by `multiply_of`.
        Loads only `category` (None |-> all).
    """
    if multiply_of is not None:
        return DirDataset(
            image_dirs=image_dir, multiply_of=multiply_of, category=category,
            shape=_SHAPE, create=download_rebelo, string=_STRING,
        )
    return DirDataset(image_dirs=image_dir, create=download_rebelo, category=category, string=_STRING)

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
