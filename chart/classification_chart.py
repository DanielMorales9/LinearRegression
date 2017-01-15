import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .chart import Chart
from utility import FeatureAugmentation as FA

class ClassificationChart(Chart):
    def __init__(self, x, y):
        super(ClassificationChart, self).__init__(x, y)

    def create_visualization(self, clf, test, z, axis1=0, axis2=1, legend=["Y", "N"]):
        # TODO To Check
        """
            Creates a 2D visualization of the training set along an axis
            for a binary classification and shows the decision boundary
                :param clf: Classifier instance
                    Classifier should already be trained
                :param test: numpy array of shape [n_samples, n_targets]
                :param axis1: int, optional
                    Index of the first axis (feature) to plot
                :param axis2: int, optional
                    Index of the second axis (feature) to plot
                :param legend: array_like, optional
                    legend of the chart
        """
        pos = np.where(self.y == 1)
        neg = np.where(self.y == 0)
        plt.scatter(self.x[pos, axis1], self.x[pos, axis2], marker='o', c='b')
        plt.scatter(self.x[neg, axis1], self.x[neg, axis2], marker='x', c='r')
        plt.xlabel("x{}".format(axis1))
        plt.ylabel("x{}".format(axis2))
        plt.legend(legend)

        cs = plt.contour(test[:, axis1], test[:, axis2], z)
        plt.clabel(cs, inline=1, fontsize=10)
