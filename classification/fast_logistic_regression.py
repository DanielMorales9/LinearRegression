import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from classification import LogisticRegression


class FastLogisticRegression(LogisticRegression):

    def __init__(self):
        super(FastLogisticRegression, self).__init__()
        self.X = None
        self.y = None
        self.l = None

    def fit(self, x, y, l=0.0):
        """
            Fits a model using L-BFGS Optimization method.

                :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                    Training data
                :param y: numpy array of shape (n_samples,)
                    Target values
                :param l: float, optional - default is zero
                    Lambda value for Regularization term
                :return x, f: ndarray, ndarray
                    Returns model params and the minimum of the cost function
        """
        self.X = self.reshape_training_set(x)
        self.y = y
        self.l = l
        initial_theta = np.zeros(self.X.shape[1])
        x, f, _ = fmin_l_bfgs_b(self.cost_function, initial_theta)

        self.model = x

    def cost_function(self, theta):
        """
            Cost function computes the model's cost and gradient
            :param theta: numpy array
                Linear Model
            :return: tuple: float, numpy array
                cost and gradient array
        """
        X = self.X
        y = self.y
        l = self.l

        z = np.dot(X, theta)
        h = 1 / (1 + np.power(np.e, -z))
        m = len(X)
        error = h - y
        grad = np.dot(X.T, error) / m
        grad[1:] += (l / m) * theta[1:]

        j = self.compute_cost(X, y, theta, l)

        return j.flatten(), grad
