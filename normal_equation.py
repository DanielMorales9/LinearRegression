import numpy as np
from numpy.linalg import inv
from normalization import *

class NormalEquation:
    """
           Normal Equation is an algorithm for Linear Regression
            Pros:
                | 1. No need to choose alpha
                | 2. Don't need to iterate

            Cons:
                | 1. Need to compute the inverse X^T * X which is O(n^3)
                | 2. Slow if number of features n is very large

       """

    def __init__(self, feature_scaling="auto"):
        self._model = None
        if feature_scaling == "mean":
            self.fs = MeanNormalization()
        elif feature_scaling == "zscore":
            self.fs = ZScoreNormalization()
        else:
            self.fs = MeanNormalization()

    def fit(self, x, y, scale_feature=True):
        """
        Fits a linear model (Theta) on training data

        :param x: numpy matrix of shape [n_samples,n_features]
            Training data
        :param y: numpy array of shape [n_samples, n_targets]
            Target values
        :param scale_feature: flag for feature scaling
        :return self:  returns an instance of self.

        """

        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)

        xn = np.ones((x.shape[0], x.shape[1] + 1))
        xn[:, 1:] = x
        xtx = np.dot(xn.T, xn)
        inverse = inv(xtx)
        self._model = np.dot(inverse, np.dot(xn.T, y))
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
            xn = np.ones(x.shape[0] + 1)
            xn[1:] = x
        return np.dot(xn, th)
