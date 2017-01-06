import numpy as np


class LinearRegression:

    def __init__(self):
        # init model
        self._model = (0.0, 0.0)
        self._diagnostics = 0

    def fit(self, x, y, alpha=0.1):
        bound = 0
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
            converged = diff0 <= bound and diff1 <= bound
            th0 = temp0
            th1 = temp1
            iteration += 1
        self._diagnostics = iteration
        self._model = th0, th1

    def predict(self, x):
        theta0, theta1 = self._model
        return theta0 + theta1 * x

    def get_diagnostics(self):
        return "Number of iterations: {}".format(self._diagnostics)