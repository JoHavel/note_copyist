import os

from datasets.dirdataset import DirDataset
from datasets.rebelo import download_rebelo, _DOWNLOADED
from datasets.rebelo2 import _rebelo_subdirs, _DEFAULT_DIR as _REB_DEFAULT_DIR
from center_images import *

_DEFAULT_DIR: str = os.path.join(_DOWNLOADED, "centered_rebelo")
""" Directory where put files of CenteredRebeloDataset, if we not provide the other directory """

_STRING: str = "crebelo"
""" String for CenteredRebeloDataset (for naming directories for models) """

FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray | None]] = {
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
""" Dictionary: symbol name -> its centering function """

REBELO_NAMES: dict[str, str] = {
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
""" Dictionary: symbol name -> its directory name in Rebelo """

_MANUALLY_DELETED_IMAGES: list[tuple[str, str]] = [('eighth-note-down', 'd0symbol3517.png'), ('eighth-note-down', 'd0symbol3851.png'), ('eighth-note-down', 'd0symbol3857.png'), ('eighth-note-down', 'd0symbol4353.png'), ('eighth-note-down', 'd0symbol4354.png'), ('eighth-note-down', 'd0symbol4368.png'), ('eighth-note-down', 'd0symbol4501.png'), ('eighth-note-down', 'd0symbol8241.png'), ('eighth-note-down', 'd1symbol10084.png'), ('eighth-note-down', 'd1symbol10567.png'), ('eighth-note-down', 'd1symbol115308.png'), ('eighth-note-down', 'd1symbol115808.png'), ('eighth-note-down', 'd1symbol116314.png'), ('eighth-note-down', 'd1symbol116823.png'), ('eighth-note-down', 'd1symbol117327.png'), ('eighth-note-down', 'd1symbol11980.png'), ('eighth-note-down', 'd1symbol119827.png'), ('eighth-note-down', 'd1symbol120326.png'), ('eighth-note-up', 'd1symbol10054.png'), ('eighth-note-up', 'd1symbol102.png'), ('eighth-note-up', 'd1symbol13281.png')]
""" We removed images of sixteenth and thirty-second notes, here is their list in the form [(symbol, centered_image_name), ...] """


def create_centered_rebelo(image_dirs: list[str] | str, output_dir: str):
    """
        Centers images from the Rebelo dataset (`image_dirs` is real and syn directory of the Rebelo dataset),
        and outputs them to `output_dir`.
    """
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
) -> DirDataset:
    """
        Dataset containing the automatically centered images of the Rebelo dataset. Those images are loaded
        from `image_dir`, we only load `category` (if it is not None) and images are padded, so it has dimensions
        divisible by `multiply_of`.

        If `image_dir` is empy we automatically center the images from the Rebelo dataset stored in `rebelo_dir`.
    """
    return DirDataset(
        image_dirs=image_dir,
        string=_STRING,
        shape=(0, 0),
        multiply_of=multiply_of,
        create=lambda: create_centered_rebelo(_rebelo_subdirs(rebelo_dir), image_dir),
        category=category,
    )
