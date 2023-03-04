# The implementation of deep convolutional GAN is omitted because of it is used for learning purposes in course NPFL114
from dcgan import *  # Implemented https://github.com/ufal/npfl114/blob/past-2122/labs/12/dcgan.py
import matplotlib.pyplot as plt

args = parser.parse_args([] if "__file__" not in globals() else None)
# args.z_dim = 2
args.z_dim = 3
args.epochs = 10

#
# code from dcgan.main
#

for bla in range(10):

    h = []
    for i in range(10):
        h.append([])
        for j in range(10):
            h[-1].append(network.generator(np.array([i/10.0, j/10.0, bla/10.0])[None])[0])

    g = []
    for i in h:
        g.append(np.concatenate(i, axis=0))

    data = np.concatenate(g, axis=1)

    # https://stackoverflow.com/questions/2659312/how-do-i-convert-a-numpy-array-to-and-display-an-image
    plt.imshow(data, interpolation='nearest')
    plt.show()
