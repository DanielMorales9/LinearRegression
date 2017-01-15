import numpy as np
from .regression_chart import RegressionChart
import matplotlib.pyplot as plt

class MultivariateLinearRegressionChart(RegressionChart):

    def __init__(self, x, y):
        super(MultivariateLinearRegressionChart, self).__init__(x, y)

    def create_convergence(self, clf, alpha, stop_alpha=None,
                           step_alpha=0.3, iterations=100):
        """
        Creates a convergence chart displaying the j value @ iteration for each
        alpha value
            :param clf: Classifier instance
                The classifier must be of type MultivariateLinearRegression
            :param alpha: float
                Start Alpha
            :param stop_alpha: float, optional
                Stop Alpha
            :param step_alpha: float, optional
                Step Alpha
            :param iterations: int, optional
                Number of iteration to train the classifier on
        """
        if stop_alpha is None:
            alphas = np.array([alpha])
        else:
            alphas = np.arange(alpha, stop_alpha, step_alpha)

        values = [clf.fit_model(self.x, self.y, a, iterations).j_history
                  for a in alphas]

        plt.figure()
        it = np.arange(iterations)

        for cost in values:
            plt.plot(it, cost, linewidth=2)

        legend = ["$\\alpha = {:.2f}$".format(a) for a in alphas]
        plt.legend(legend)
        plt.ylabel("Cost values")
        plt.xlabel("Iterations")
        plt.title('Convergence Chart')