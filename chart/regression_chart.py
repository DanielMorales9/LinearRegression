import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .chart import Chart

class RegressionChart(Chart):

    def __init__(self, x, y):
        super(RegressionChart, self).__init__(x, y)

    def create_visualization3D(self, clf, test, axis1=0, axis2=1):
        """
        Creates the visualization of the data along an axis (feature)
            :param clf: Classifier instance
                The classifier must be a regressor and should already be trained
            :param test: numpy array of shape [n_samples, n_targets]
            :param axis1: int, optional
                Index of the first axis (feature) to plot
            :param axis2: int, optional
                Index of the second axis (feature) to plot
            :param test: numpy matrix of shape [n_samples,n_features]
                Test data to plot
        """

        pred = clf.predict(test)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(test[:, axis1], test[:, axis2])
        ax.plot_surface(x, y, pred,
                        rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)

        ax.scatter(self.x[:, 0], self.x[:, 1], self.y, c='k', marker='o')
        plt.xlabel('x{}'.format(axis1))
        plt.ylabel('x{}'.format(axis2))
        ax.set_zlabel('Prediction')

        plt.title('Surface plot')

    def create_visualization(self, clf, test, axis=0):
        """
        Creates a 2D visualization of the training set along an axis
        And shows the prediction for test
            :param clf: Classifier instance
                Classifier should already be trained
            :param test: numpy matrix of shape [n_samples,n_features]
                Test data to plot
            :param axis: int, optional
                Index of the axis (feature) to plot
        """
        pred = clf.predict(test)
        plt.figure()
        plt.plot(self.x[:, axis], self.y, 'rx')
        plt.plot(test[:, axis], pred)
        plt.xlabel("x{}".format(axis))
        plt.ylabel("Target")
