import numpy as np
from linear_regression import LinearRegression


class MultivariateLinearRegression(LinearRegression):

    def fit(self, x, y, alpha=0.1):
        xn = np.ones((x.shape[0], x.shape[1]+1))
        xn[:, 1:] = x

        th = np.zeros(xn.shape[1])
        m = len(x)
        bound = 0
        beta = alpha / m
        iteration = 0
        converged = False
        while not converged:
            error = np.dot(xn, th) - y
            temp = th - beta * np.dot(error.T, xn)
            diff = abs(temp - th) <= bound
            converged = np.all(diff)
            th = temp
            iteration += 1
        self._diagnostics = iteration
        self._model = th

    def predict(self, x):
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
        else:
            xn = np.ones(x.shape[0]+1)
            xn[1:] = x
        return np.dot(xn, th)
