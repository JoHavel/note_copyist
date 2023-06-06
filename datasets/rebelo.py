from datasets.dirdataset import DirDataset

_SHAPE = (20, 20)
_DEFAULT_DIR = "./downloaded/Rebelo Dataset/MainClasses"

def RebeloDataset(
        image_dir: str = _DEFAULT_DIR,
        multiply_of: int | None = None,
):
    if multiply_of is not None:
        return DirDataset(image_dirs=image_dir, multiply_of=multiply_of, shape=_SHAPE)
    return DirDataset(image_dirs=image_dir)

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
