import numpy as np

from Assignment_2.sigmoid import sigmoid


def predict_value(example, hypothesis):
    value = np.dot(example, hypothesis)
    prediction = sigmoid(value)
    return prediction

