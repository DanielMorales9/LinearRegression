import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from regression import MultivariateLinearRegression as mlr


class Chart(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def create_convergence(self, clf, alpha, stop_alpha=None,
                           step_alpha=0.3, iterations=100):
        if stop_alpha is None:
            alphas = np.array([alpha])
        else:
            alphas = np.arange(alpha, stop_alpha, step_alpha)

        values = [clf.fit_model(self.X, self.y, a, iterations).j_history
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

    def create_surface(self, clf, theta0, theta1):
        j, x, y = self._compute_cost_by_theta(clf, theta0, theta1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, j,
                        rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)

        plt.xlabel('theta 0')
        plt.ylabel('theta 1')
        plt.title('Surface plot')

    def create_contour(self, clf, theta0, theta1):
        z, x, y = self._compute_cost_by_theta(clf, theta0, theta1)

        plt.figure()
        cs = plt.contour(x, y, z)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.xlabel(r"${\Theta}_0$")
        plt.ylabel(r"${\Theta}_1$")
        plt.title('Contour plot')

    def create_visualization(self, clf, test, axis=0):
        """
        Creates the visualization of the data along an axis (feature)
            :param clf: Classifier instance
                The classifier must be a regressor and should already be trained
            :param test: numpy array of shape [n_samples, n_targets]
            :param axis: int
                Index of the axis (feature) to plot
            :param test: numpy matrix of shape [n_samples,n_features]
                Test data to plot

        """
        pred = clf.predict(test)
        plt.figure()
        plt.plot(self.X[:, axis], self.y, 'rx')
        plt.plot(test[:, axis], pred)
        plt.xlabel("x{}".format(axis))
        plt.ylabel("Target")

    def show(self):
        plt.show()

    def _compute_cost_by_theta(self, clf, theta0, theta1):
        X = mlr.reshape_training_set(self.X)

        th0 = X[:, theta0]
        min0, max0 = min(th0), max(th0)
        th1 = X[:, theta1]
        min1, max1 = min(th1), max(th1)

        r0 = max0 - min0
        if r0 == 0:
            step0 = max0
            min0 = 0
        else:
            step0 = r0
        step0 /= 10

        r1 = max1 - min1
        if r1 == 0:
            step1 = max1
            min1 = 0
        else:
            step1 = r1
        step1 /= 10

        x = np.arange(min0, max0, step0)
        y = np.arange(min1, max1, step1)

        z = np.zeros((len(x), len(y)))
        in0 = np.arange(0, len(x))
        in1 = np.arange(0, len(y))
        theta = np.zeros(X.shape[1])
        for i in in0:
            for k in in1:
                theta[theta0] = x[i]
                theta[theta1] = y[k]
                z[i, k] = clf.compute_cost(X, self.y, theta)

        x, y = np.meshgrid(x, y)
        return z, x, y