import numpy as np


class LinearRegression:
    """
        Linear Regression with Gradient Descent
            Pros:
                | 1. Scale with number of features n

            Cons:
                | 1. Need to choose alpha
                | 2. Need to iterate

          """

    def __init__(self):
        # init model
        self._model = (0.0, 0.0)
        self._diagnostics = 0

    def fit(self, x, y, alpha=0.1, tol=0):
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
        :return self:  returns an instance of self.

        """
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

    def predict(self, x):
        """
        Predict using the linear model

        :param x: {array-like}, shape = (n_samples,)
            Samples.
        :return: target: array, shape = (n_samples,)
            Returns predicted values.
        """
        theta0, theta1 = self._model
        return theta0 + theta1 * x

    def get_diagnostics(self):
        return "Number of iterations: {}".format(self._diagnostics)