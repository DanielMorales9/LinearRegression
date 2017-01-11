import numpy as np
from numpy.linalg import inv
from .linear_regression import LinearRegression


class NormalEquation(LinearRegression):
    """
    Normal Equation is an algorithm for Linear Regression
        Pros:
            | 1. No need to choose alpha
            | 2. Don't need to iterate
            | 3. No need for Feature Scaling

        Cons:
             | 1. Need to compute the inverse X^T * X which is O(n^3)
             | 2. Slow if number of features n is very large
    """

    def __init__(self):
        super(NormalEquation, self).__init__()

    def fit(self, x, y):
        """
        Fits a regression model (Theta) on training data

        :param x: numpy matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values

        :return self:  returns an instance of self.

        """

        xn = self.reshape_training_set(x)
        xtx = np.dot(xn.T, xn)
        inverse = inv(xtx)
        self._model = np.dot(inverse, np.dot(xn.T, y))
        return self

    def predict(self, x):
        """
        Predict using the regression model

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                Samples.
            :return: target: array, shape = (n_samples,)
                Returns predicted values.
        """
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
        else:
            xn = np.ones(x.shape[0] + 1)
            xn[1:] = x
        return np.dot(xn, th)