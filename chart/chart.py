import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Chart(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def create_contour(self, clf, theta0, theta1):
        z, x, y = self._compute_cost_by_theta(clf, theta0, theta1)

        plt.figure()
        cs = plt.contour(x, y, z)
        plt.clabel(cs, inline=1, fontsize=10)
        plt.xlabel(r"${\Theta}_0$")
        plt.ylabel(r"${\Theta}_1$")
        plt.title('Contour plot')

    def create_error_surface(self, clf, theta0=0, theta1=1):
        """
        Creates Error Surface and displays it in the axis theta1 and theta2
            :param clf: Classifier Instance
                The Classifier will compute cost when theta0 and theta1 changes
            :param theta0: int, optional
                Index of the axis (feature) to display
            :param theta1: int, optional
                Index of the axis (feature) to display
        """
        j, x, y = self._compute_cost_by_theta(clf, theta0, theta1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(x, y, j,
                        rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)

        plt.xlabel('theta 0')
        plt.ylabel('theta 1')
        plt.title('Surface plot')

    def show(self):
        """
        Shows all the figure created before.
        """
        plt.show()

    def _compute_cost_by_theta(self, clf, theta0, theta1):
        X = clf.reshape_training_set(self.x)

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