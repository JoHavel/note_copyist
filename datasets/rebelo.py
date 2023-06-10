import os

from datasets.dirdataset import DirDataset

_SHAPE = (20, 20)

_URL = "https://github.com/apacha/OMR-Datasets/releases/download/datasets/Rebelo.Dataset.zip"
_DOWNLOADED = "./downloaded"
_ZIP_FILE = os.path.join(_DOWNLOADED, "rebelo.zip")
_DATASET = os.path.join(_DOWNLOADED, "Rebelo Dataset")
_7ZIP_FILES = [os.path.join(_DATASET, "database1.7z"), os.path.join(_DATASET, "database2.7z"), os.path.join(_DATASET, "database3.7z")]
_DEFAULT_DIR = os.path.join(_DATASET, "MainClasses")


def download_rebelo():
    from sys import stderr
    from requests import get
    from zipfile import ZipFile
    from py7zr import SevenZipFile

    if not os.path.exists(_DOWNLOADED):
        os.mkdir(_DOWNLOADED)

    print("Downloading Rebelo dataset.", file=stderr)

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
):
    if multiply_of is not None:
        return DirDataset(image_dirs=image_dir, multiply_of=multiply_of, shape=_SHAPE, create=download_rebelo)
    return DirDataset(image_dirs=image_dir, create=download_rebelo)

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
