import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# דוגמה לשימוש:
z_scalar = 0.5
z_vector = np.array([0.5, 1.0, 1.5])
z_matrix = np.array([[0.5, 1.0], [1.5, 2.0]])

# ערכים מספרים ממשיים גדולים - צפוי קרוב ל-1
large_positive_values = sigmoid(z_matrix)
print(large_positive_values)

# ערכים מספרים שליליים גדולים - צפוי קרוב ל-0
large_negative_values = sigmoid(-z_matrix)
print(large_negative_values)
