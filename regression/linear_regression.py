import numpy as np
from abc import ABCMeta, abstractmethod

class LinearRegression(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._model = None
        self.j_history = None

    @abstractmethod
    def fit(self, x, y):
        """
        Fits a regression model (Theta) on training data

        :param x: numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        :param y: numpy array of shape (n_samples,)
            Target values
        :return self:  returns an instance of self.
    """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Predict using the regression model

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                Samples.
            :return: target: array, shape = (n_samples,)
                Returns predicted values.
        """
        pass

    @staticmethod
    def reshape_training_set(x):
        """
        Returns training set with a column of ones
            :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                  Training data
            :return: x_new: numpy array or sparse matrix of shape [n_samples, n_features+1]
                  Training set with one more column of ones
        """
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1]+1))
            xn[:, 1:] = x
        else:
            xn = np.ones((x.shape[0], 2))
            xn[:, 1] = x

        return xn

    @staticmethod
    def compute_cost(X, y, theta):
        """
        Compute the Cost J given theta and training data
            :param X: numpy array or sparse matrix of shape [n_samples, n_features]
                Training data
            :param y: numpy array of shape (n_samples,)
                Target values
            :param theta: numpy array of shape(n_samples,)
                Linear Model
            :return: j: float
                Least Mean Square Cost
        """

        m = len(X)
        h = np.dot(X, theta) - y
        h **= 2
        error = np.sum(h)
        error /= 2 * m
        return error
