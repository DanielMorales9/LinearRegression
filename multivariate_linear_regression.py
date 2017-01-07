import numpy as np
from linear_regression import LinearRegression
from normalization import *


class MultivariateLinearRegression(LinearRegression):
    """
        Multivariate Linear Regression with Gradient Descent
            Pros:
                | 1. Scale with number of features n

            Cons:
                | 1. Need to choose alpha
                | 2. Need to iterate
    """

    def __init__(self, feature_scaling="auto"):
        self.j_value = []
        if feature_scaling == "mean":
            self.fs = MeanNormalization()
        elif feature_scaling == "zscore":
            self.fs = ZScoreNormalization()
        else:
            self.fs = MeanNormalization()

    def fit(self, x, y, alpha=0.1, tol=0, scale_feature=True):
        """
              Fits a linear model (Theta) on training data

              :param x: numpy array or sparse matrix of shape [n_samples, n_features]
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

        xn = np.ones((x.shape[0], x.shape[1]+1))
        xn[:, 1:] = x

        th = np.zeros(xn.shape[1])
        m = len(x)
        beta = alpha / m
        iteration = 0
        converged = False
        while not converged:
            error = np.dot(xn, th) - y
            # self.pick_value_of_j(np.sum(error**2)*beta, iteration)
            temp = th - beta * np.dot(error.T, xn)
            diff = abs(temp - th) <= tol
            converged = np.all(diff)
            th = temp
            iteration += 1
        self._diagnostics = iteration
        self._model = th
        return self

    def predict(self, x, scale_feature=True):
        """
        Predict using the linear model

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                  Samples.
            :param scale_feature: flag for feature scaling
            :return: target: array, shape = (n_samples,)
                  Returns predicted values.

        """
        if scale_feature:
            x = self.fs.scale(x)
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
        else:
            xn = np.ones(x.shape[0]+1)
            xn[1:] = x
        return np.dot(xn, th)

    def pick_value_of_j(self, j, iteration):
        self.j_value.append([j, iteration])
