import numpy as np
import matplotlib.pyplot as plt

from Assignment_2.addOnesColumn import addOnesColumn
from Assignment_2.computeCostAndGradient import computeCostAndGradient
from Assignment_2.loadData import load_data
from Assignment_2.plotDecisionBoundary import plotDecisionBoundary
from Assignment_2.updateHypothsis import updateHypothesis


def gradientDescent(filename, alpha, max_iter, threshold):
    D, Y = load_data(filename)
    Data = addOnesColumn(D)

    # Initialize
    i = 0
    flag = True
    Hypothesis = np.array([-8, 2, -0.5])

    Costs = []

    while flag:
        i += 1

        compCost, gradient = computeCostAndGradient(Data, Y, Hypothesis)

        # Cost
        Costs.append(compCost)

        Hypothesis = updateHypothesis(Hypothesis, alpha, gradient)

        # Improvement
        if i >= 2 and (abs(Costs[i - 2] - Costs[i - 1]) <= threshold):
            flag = False
            print(f"Gradient descent terminating after {i} iterations. "
                  f"Improvement was: {Costs[i - 2] - Costs[i - 1]} below threshold ({threshold})")

        # Max iterations
        if i > max_iter:
            flag = False
            print(f"Gradient descent terminating after {max_iter} iterations (max_iter)")

    # Update
    FinalHypothesis = updateHypothesis(Hypothesis, alpha, gradient)

    # Graph cost
    plt.plot(Costs)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over iterations')
    plt.show()

    # Plot decision boundary
    plotDecisionBoundary(FinalHypothesis, Data, Y)

    return Costs, FinalHypothesis