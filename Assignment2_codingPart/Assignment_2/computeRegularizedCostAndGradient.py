import numpy as np

from Assignment_2.predictValue import predict_value


def computeRegularizedCostAndGradient(D, Y, Hypothesis, lambda_val):
    m, n = D.shape  # גודל הנתונים

    Gradient = np.zeros(n)  # יצירת מטריצת גרדיאנט ריקה
    J = 0  # יצירת משתנה עבור עלות

    eps = 0.0001  # קצה שגיאה קטן

    for i in range(m):
        predictVal = predict_value(np.array([D[i, :]]), Hypothesis)  # חישוב הניבוי
        if predictVal == 0:
            J += (-Y[i] * np.log(predictVal + eps) - (1 - Y[i]) * np.log(1 - predictVal))
        elif predictVal == 1:
            J += (-Y[i] * np.log(predictVal) - (1 - Y[i]) * np.log(1 - (predictVal - eps)))
        else:
            J += (-Y[i] * np.log(predictVal) - (1 - Y[i]) * np.log(1 - predictVal))

        # חישוב הגרדיאנט
        for j in range(n):
            error = predictVal - Y[i]
            Gradient[j] += error * D[i, j]

    # חישוב עבור הרגולריזציה
    theta = np.sum(Hypothesis[1:] ** 2)  # הסכום הריבועי של פרמטרי היפותזה השארים
    J = J / m + (lambda_val / (2 * m)) * theta

    # חישוב הגרדיאנט לרגולריזציה
    gradTheta = np.sum(Hypothesis[1:])  # סכום ערכי ההיפותזה למעט המקדם של הרגולריזציה
    Gradient[1:] = (Gradient[1:] / m) + (lambda_val / m) * Hypothesis[1:]

    return J, Gradient


# טענת הדרך
D = np.array([[1, 2, 3], [4, 5, 6]])
Y = np.array([0, 1])
Hypothesis = np.array([-10, 0.8, 0.08])
lambda_val = 0.0001

# בדיקה
J, Gradient = computeRegularizedCostAndGradient(D, Y, Hypothesis, lambda_val)
print("Cost J:", J)
print("Gradient:", Gradient)
