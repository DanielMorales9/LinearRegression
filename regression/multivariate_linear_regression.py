import numpy as np
from normalization import *
from .linear_regression import LinearRegression


class MultivariateLinearRegression(LinearRegression):
    """
        Multivariate Linear Regression with Gradient Descent
            Pros:
                | 1. Scale with number of features n

            Cons:
                | 1. Need to choose alpha
                | 2. Need to iterate
                | 3. Need Feature Scaling
    """

    def __init__(self, feature_scaling="auto"):
        super(MultivariateLinearRegression, self).__init__()
        if feature_scaling == "mean":
            self.fs = MeanNormalization()
        elif feature_scaling == "zscore":
            self.fs = ZScoreNormalization()
        else:
            self.fs = MeanNormalization()

    def fit(self, x, y, alpha=0.1, l=0, tol=0, scale_feature=True):
        """
              Fits a regression model (Theta) on training data

              :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                    Training data
              :param y: numpy array of shape (n_samples,)
                    Target values
              :param alpha: float, default: 0.1
                    Learning rate
              :param l: float, optional - default is zero
                    Lambda value for Regularization term
              :param tol: float, default: 0
                    Tollerance of convergence value
              :param scale_feature: boolean, optional
                    Flag for feature scaling, default is True
              :return self:  returns an instance of self.

        """


        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)
        x = self.reshape_training_set(x)

        th = np.zeros(x.shape[1])
        m = len(x)
        beta = alpha / m
        converged = False
        while not converged:
            error = np.dot(x, th) - y
            temp = th - beta * np.dot(error.T, x)
            temp[1:] += l * beta * th[1:]
            diff = abs(temp - th) <= tol
            converged = np.all(diff)

            th = temp

        self._model = th

        return self

    def fit_model(self, x, y, alpha, iterations, l=0.0, scale_feature=True):
        """
            Fits a regression model (Theta) on training data

              :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                  Training data
              :param y: numpy array of shape (n_samples,)
                  Target values
              :param alpha: float
                  Learning rate
              :param iterations: int
                  number of iterations
              :param l: float, optional - default is zero
                  Lambda value for Regularization term
              :param scale_feature: flag for feature scaling
              :return self:  returns an instance of self.

              """

        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)

        x = self.reshape_training_set(x)

        th = np.zeros(x.shape[1])
        m = len(x)
        beta = alpha / m
        i = 0

        j_history = np.zeros(iterations)

        while i < iterations:
            
            error = np.dot(x, th) - y
            th[1:] = (1 - l * beta) * th[1:]
            th -= beta * np.dot(error.T, x)
            j_history[i] = self.compute_cost(x, y, th, l)
            i += 1

        self.j_history = j_history
        self._model = th

        return self

    def predict(self, x, scale_feature=True):
        """
        Predict using the regression model

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
