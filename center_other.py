from center_images import *


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
    "whole-note": center_whole_note,
}
""" Dictionary: symbol name -> its centering function """

OTHER_NAMES: dict[str, str] = {
    "sharp": "Sharp",
    "flat": "Flat",
    "natural": "Natural",
    "g-clef": "G-Clef",
    "f-clef": "F-Clef",
    "c-clef": "C-Clef",
    "half-note": "Half-Note",
    "quarter-note": "Quarter-Note",
    "eighth-note-up": "Eighth-Note",
    "eighth-note-down": "Eighth-Note",
    "quarter-rest": "Quarter-Rest",
    "whole-note": "Whole-Note",
}
""" Dictionary: symbol name -> its directory name in other dataset """


def create_centered_other(image_dirs: list[str] | str = "downloaded/others", output_dir: str = "downloaded/other"):
    """
        Centers images from the custom dataset in `image_dirs`, and outputs them to `output_dir`.
    """
    from sys import stderr
    print("Centering `other` dataset.", file=stderr)
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    if not all([os.path.exists(dirr) for dirr in image_dirs]):
        raise Exception("Some directory does not exists: " + str(image_dirs))

    os.mkdir(output_dir)

    for symbol, reb_directory in OTHER_NAMES.items():
        symbol_output_dir = os.path.join(output_dir, symbol)
        dirs = list(filter(os.path.exists, [os.path.join(dirr, reb_directory) for dirr in image_dirs]))
        assert len(dirs) != 0
        center_images(dirs, symbol_output_dir, FUNCTIONS[symbol])

    print("`other` dataset centered.", file=stderr)


if __name__ == '__main__':
    create_centered_other()