import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from regression import MultivariateLinearRegression as MGD


class LinearRegressionChart(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def create_convergence(self, alpha, stop_alpha=None, step_alpha=0.3, iterations=100):
        if stop_alpha is None:
            alphas = np.array([alpha])
        else:
            alphas = np.arange(alpha, stop_alpha, step_alpha)

        clf = MGD()
        values = \
            [clf.fit(self.X, self.y, a, iterations).j_history for a in alphas]

        plt.figure()
        it = np.arange(iterations)

        for cost in values:
            plt.plot(it, cost, linewidth=2)

        legend = ["$\\alpha = {:.2f}$".format(a) for a in alphas]
        plt.legend(legend)
        plt.ylabel("Cost values")
        plt.xlabel("Iterations")
        plt.title('Convergence Chart')

    def create_surface(self, theta0, theta1):
        j, x, y = self._compute_cost_by_theta(theta0, theta1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, j,
                        rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)

        plt.xlabel('theta 0')
        plt.ylabel('theta 1')
        plt.title('Surface plot')

    def create_contour(self, theta0, theta1):
        z, x, y = self._compute_cost_by_theta(theta0, theta1)

        plt.figure()
        cs = plt.contour(x, y, z)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.xlabel(r"${\Theta}_0$")
        plt.ylabel(r"${\Theta}_1$")
        plt.title('Contour plot')

    def show(self):
        plt.show()

    def _compute_cost_by_theta(self, theta0, theta1):
        X = MGD.reshape_training_set(self.X)
        x = np.arange(-5, 5, 0.25)
        y = np.arange(-5, 5, 0.25)
        z = np.zeros((len(x), len(y)))
        in0 = np.arange(0, len(x))
        in1 = np.arange(0, len(y))
        theta = np.zeros(X.shape[1])
        for i in in0:
            for k in in1:
                theta[theta0] = x[i]
                theta[theta1] = y[k]
                z[i, k] = MGD.compute_cost(X, self.y, theta)

        x, y = np.meshgrid(x, y)
        return z, x, y