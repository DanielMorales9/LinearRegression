import numpy as np
from linear_regression import LinearRegression


class MultivariateLinearRegression(LinearRegression):
    """
        Multivariate Linear Regression with Gradient Descent
            Pros:
                | 1. Scale with number of features n

            Cons:
                | 1. Need to choose alpha
                | 2. Need to iterate
    """

    def __init__(self):
        super(MultivariateLinearRegression, self).__init__()
        self.j_value = []

    def fit(self, x, y, alpha=0.1, tol=0):
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
              :return self:  returns an instance of self.

              """
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

    def predict(self, x):
        """
        Predict using the linear model

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                  Samples.
            :param scale_feature: flag for feature scaling
            :return: target: array, shape = (n_samples,)
                  Returns predicted values.

        """
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
