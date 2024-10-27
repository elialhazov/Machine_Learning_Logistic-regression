import numpy as np


def updateHypothesis(Hypothesis, alpha, Gradient):
    UpdatedHypothesis = np.zeros_like(Hypothesis) #מערך של אפסים באותו גודל
    for i in range(len(Hypothesis)):
        UpdatedHypothesis[i] = Hypothesis[i] - (alpha * Gradient[i])
    return UpdatedHypothesis
