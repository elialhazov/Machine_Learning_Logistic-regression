import numpy as np


def load_data(filename):
    # load file
    file = np.loadtxt(filename)
    col = file.shape[1]
    D = file[:, :col - 1]
    Y = file[:, col - 1]
    return D, Y
