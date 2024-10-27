import numpy as np

from Assignment_2.predictValue import predict_value


def computeCostAndGradient(D, Y, Hypothesis):
    # גודל הנתונים
    m = D.shape[0]

    # גודל ההיפותזה
    num_features = Hypothesis.shape[1]

    # אתחול סכום המחיר והגרדיאנט
    J = 0
    Gradient = np.zeros((1, num_features))

    # לולאה על שורות הנתונים
    for i in range(m):
        # היפותזה עבור הנתון הנוכחי
        prediction = predict_value(Hypothesis[i], D[i])

        # חישוב המחיר והוספתו לסך הכל של המחירים
        J += Y[i] * np.log(prediction) + (1 - Y[i]) * np.log(1 - prediction)

        # חישוב השגיאה
        error = prediction - Y[i]

        # עדכון הגרדיאנט
        Gradient += error * D[i]

    # חישוב הממוצע של המחיר
    J = -J / m

    # חישוב הממוצע של הגרדיאנט
    Gradient = Gradient / m

    return J, Gradient
