import numpy as np
import matplotlib.pyplot as plt
from fontTools.mtiLib import mapFeature


def plotDecisionBoundary(theta, X, y):
    # Plot Data
    positive = np.where(y == 1)
    negative = np.where(y == 0)
    plt.plot(X[positive, 0], X[positive, 1], 'k+', markersize=7, label='Positive')
    plt.plot(X[negative, 0], X[negative, 1], 'yo', markersize=7, label='Negative')

    if theta[0] == 0:
        theta[0] = 0.001
    if theta[1] == 0:
        theta[1] = 0.001
    if theta[2] == 0:
        theta[2] = 0.001

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y, 'b-', label='Decision Boundary')
        plt.legend(loc='upper right')
        plt.axis([20, 100, 20, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j]).dot(theta)

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, levels=[0], linewidths=2)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Decision Boundary')
    plt.show()


# Example usage
# Replace the below values with your actual data
theta = np.array([0.5, 0.5, 0.5])
X = np.array([[1, 20, 30],
              [1, 40, 50],
              [1, 60, 70]])
y = np.array([0, 1, 0])

plotDecisionBoundary(theta, X, y)
