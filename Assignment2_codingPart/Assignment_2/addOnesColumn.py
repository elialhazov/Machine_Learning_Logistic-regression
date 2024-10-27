import numpy as np


def addOnesColumn(D):
    rows, cols = D.shape
    colOnes = np.ones((rows, 1))
    return np.hstack((colOnes, D))
