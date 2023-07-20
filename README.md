# NoteCopyist

## Task

The task is to generate pieces of note notation for <https://github.com/Jirka-Mayer/Mashcima>
to obtain more variable synthetic train data for Optical Music Recognition (OMR).

## Using NoteCopyist
After installing [`requirements.txt`](requirements.txt), we can directly run `python main.py`.
The command `python main.py --help` explains the arguments. However, a good start is,
for example:
```shell
python main.py --dataset crebelo --cat onecat --layers --conv_layers 64 16 4 --stride 3 --kernel 5 --multiply_of 27 --batch 5 --network aae --epoch 150 --latent 4
```

The parameter `dataset` can be `mnist` for the MNIST dataset, `rebelo1` for the square
Rebelo dataset, `rebelo2` for the original-sized Rebelo dataset, `crebelo` for rebelo2
automatically centered to the attachment points, and other for custom dataset (divided to
directories by category) in directory `downloaded/other/`.

The parameter `cat` (a category) can be `basic` for unlabeled training, `onecat` also for
unlabeled training, but for every category separately, or `cat` for category-conditioned
training.

It trains the desired generative NN and creates a deep directory tree in the directory
out. The deepest directories are images, where previews for each epoch and category
are stored, and parts where the models are saved.[^1]

[^1]: The models can be loaded in Python by generators.\*\*\*.\*\*\*.load_all(filename).

Then we can generate the images (and text files with the positions of the uncropped
images’ centers) by running:
```shell
python generate_images.py out/{dataset}/{cat}/{NN description}/parts/e{epoch} {output_dir} {number} --network {aae/vae/gan}
```

Finally, we need to add the other end of the stem to every image of a stemmed note.
The following command serves this purpose (it adds a text file with the coordinates of
stems’ ends to every image in directories provided as arguments):
```shell
python add_other_end_of_stem.py {output_dir}/half-note {output_dir}/quarter-note {output_dir}/eighth-note-up {output_dir}/eighth-note-down
```

The other end of the stems works only for the normalized (crebelo) images.

