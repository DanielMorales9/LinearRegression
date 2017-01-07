import numpy as np
from normalization import *

class LinearRegression:
    """
        Linear Regression with Gradient Descent
            Pros:
                | 1. Scale with number of features n

            Cons:
                | 1. Need to choose alpha
                | 2. Need to iterate

          """

    def __init__(self, feature_scaling="auto"):
        # init model
        self._model = (0.0, 0.0)
        self._diagnostics = 0
        if feature_scaling == "mean":
            self.fs = MeanNormalization()
        elif feature_scaling == "zscore":
            self.fs = ZScoreNormalization()
        else:
            self.fs = MeanNormalization()

    def fit(self, x, y, alpha=0.1, tol=0, scale_feature=True):
        """
        Fits a linear model (Theta) on training data

        :param x: numpy array of shape (n_samples,)
            Training data
        :param y: numpy array of shape (n_samples,)
            Target values
        :param alpha: float
            Learning rate
        :param tol: float
            tollerance of convergence value
        :param scale_feature: flag for feature scaling
        :return self:  returns an instance of self.

        """

        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)

        th0 = 0.0
        th1 = 0.0
        m = len(x)
        beta = alpha / m
        iteration = 0
        converged = False
        while not converged:
            error = (th0 + th1 * x) - y
            temp0 = th0 - np.sum(error) * beta
            temp1 = th1 - np.sum(error * x) * beta
            diff0 = abs(temp0 - th0)
            diff1 = abs(temp1 - th1)
            converged = diff0 <= tol and diff1 <= tol
            th0 = temp0
            th1 = temp1
            iteration += 1
        self._diagnostics = iteration
        self._model = th0, th1
        return self

    def predict(self, x, scale_feature=True):
        """
        Predict using the linear model

        :param x: {array-like}, shape = (n_samples,)
            Samples.
        :param scale_feature: flag for feature scaling
        :return: target: array, shape = (n_samples,)
            Returns predicted values.
        """
        if scale_feature:
            x = self.fs.scale(x)
        theta0, theta1 = self._model
        return theta0 + theta1 * x

    def get_diagnostics(self):
        return "Number of iterations: {}".format(self._diagnostics)