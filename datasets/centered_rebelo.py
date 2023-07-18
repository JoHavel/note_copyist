import os

from datasets.dirdataset import DirDataset
from datasets.rebelo import download_rebelo
from datasets.rebelo2 import _rebelo_subdirs, _DEFAULT_DIR as _REB_DEFAULT_DIR
from center_images import *

_DOWNLOADED = "./downloaded"
_DEFAULT_DIR = os.path.join(_DOWNLOADED, "centered_rebelo")

_STRING = "crebelo"

FUNCTIONS = {
    "sharp": center_sharp,
    "flat": center_flat,
    "natural": center_natural,
    "g-clef": center_g_clef,
    "f-clef": center_f_clef,
    "c-clef": center_c_clef,
    "half-note": center_half_note,
    "quarter-note": center_quarter_note,
    "eighth-note-up": center_eighth_note_up,
    "eighth-note-down": center_eighth_note_down,
    "quarter-rest": center_quarter_rest,
}

REBELO_NAMES = {
    "sharp": "sharps",
    "flat": "flat",
    "natural": "naturals",
    "g-clef": "trebleClef",
    "f-clef": "bassClef",
    "c-clef": "altoClef",
    "half-note": "notesOpen",
    "quarter-note": "notes",
    "eighth-note-up": "notesFlags",
    "eighth-note-down": "notesFlags",
    "quarter-rest": "rests1",
}

_MANUALLY_DELETED_IMAGES = [('eighth-note-down', 'd0symbol3517.png'), ('eighth-note-down', 'd0symbol3851.png'), ('eighth-note-down', 'd0symbol3857.png'), ('eighth-note-down', 'd0symbol4353.png'), ('eighth-note-down', 'd0symbol4354.png'), ('eighth-note-down', 'd0symbol4368.png'), ('eighth-note-down', 'd0symbol4501.png'), ('eighth-note-down', 'd0symbol8241.png'), ('eighth-note-down', 'd1symbol10084.png'), ('eighth-note-down', 'd1symbol10567.png'), ('eighth-note-down', 'd1symbol115308.png'), ('eighth-note-down', 'd1symbol115808.png'), ('eighth-note-down', 'd1symbol116314.png'), ('eighth-note-down', 'd1symbol116823.png'), ('eighth-note-down', 'd1symbol117327.png'), ('eighth-note-down', 'd1symbol11980.png'), ('eighth-note-down', 'd1symbol119827.png'), ('eighth-note-down', 'd1symbol120326.png'), ('eighth-note-up', 'd1symbol10054.png'), ('eighth-note-up', 'd1symbol102.png'), ('eighth-note-up', 'd1symbol13281.png')]


def create_centered_rebelo(image_dirs: list[str] | str, output_dir: str):
    from sys import stderr
    print("Centering Rebelo dataset.", file=stderr)
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    if not all([os.path.exists(dirr) for dirr in image_dirs]):
        download_rebelo()

    os.mkdir(output_dir)

    for symbol, reb_directory in REBELO_NAMES.items():
        symbol_output_dir = os.path.join(output_dir, symbol)
        dirs = list(filter(os.path.exists, [os.path.join(dirr, reb_directory) for dirr in image_dirs]))
        assert len(dirs) != 0
        center_images(dirs, symbol_output_dir, FUNCTIONS[symbol])


    symbol = "whole-note"
    symbol_output_dir = os.path.join(output_dir, symbol)
    os.mkdir(symbol_output_dir)
    dirs = list(filter(os.path.exists, [os.path.join(dirr, REBELO_NAMES["half-note"]) for dirr in image_dirs]))
    assert len(dirs) != 0
    center_images(dirs, symbol_output_dir, whole_note_from_half)


    for symbol, image in _MANUALLY_DELETED_IMAGES:
        os.remove(os.path.join(output_dir, symbol, image))

    print("Rebelo dataset centered.", file=stderr)


def CenteredRebeloDataset(
        category: str | int = None,
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
        rebelo_dir: str = _REB_DEFAULT_DIR
):
    return DirDataset(
        image_dirs=image_dir,
        string=_STRING,
        shape=(0, 0),
        multiply_of=multiply_of,
        create=lambda: create_centered_rebelo(_rebelo_subdirs(rebelo_dir), image_dir),
        category=category,
    )
