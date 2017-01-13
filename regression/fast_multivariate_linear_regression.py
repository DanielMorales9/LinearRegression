import numpy as np
from regression import MultivariateLinearRegression
from scipy.optimize import fmin_l_bfgs_b


class FastMultivariateLinearRegression(MultivariateLinearRegression):

    def __init__(self):
        super(FastMultivariateLinearRegression, self).__init__()
        self.X = None
        self.y = None
        self.l = None

    def fit(self, x, y, l=0, scale_feature=True):
        """
            Fits a linear model using L-BFGS Optimization method.

            :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                Training data
            :param y: numpy array of shape (n_samples,)
                Target values
            :param l: float, optional - default is zero
                Lambda value for Regularization term
            :param scale_feature: boolean, optional
                Flag for feature scaling, default is True
            :return x, f: ndarray, ndarray
                Returns model params and the minimum of the cost function
        """
        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)
        self.X = self.reshape_training_set(x)
        self.y = y
        self.l = l
        initial_theta = np.zeros(self.X.shape[1])
        x, f, _ = fmin_l_bfgs_b(self.cost_function, initial_theta)

        self.model = x
        return x, f

    def cost_function(self, theta):
        X = self.X
        y = self.y
        l = self.l

        m = len(X)

        error = np.dot(X, theta) - y
        grad = np.dot(error.T, X) / m
        grad[1:] += (1 - l * m) * theta[1:]
        j = self.compute_cost(X, y, theta, l)
        return j, grad
