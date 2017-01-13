import numpy as np


class LogisticRegression(object):
    """
    Logistic Regression is a classification algorithm based on gradient descent

    """

    def __init__(self):
        self._model = None
        self._j_history = None

    def fit(self, x, y, alpha=0.1, l=0.0, tol=0.0):
        """
        Fits a regression model (Theta) on training data

            :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                Training data
            :param y: numpy array of shape (n_samples,)
                Target values
            :param alpha: float
                Learning rate
            :param l: float, optional - default is zero
                Lambda value for Regularization term
            :param tol: float
                Tollerance of convergence value
            :return self:
                Returns an instance of self.

        """

        x = self.reshape_training_set(x)

        th = np.zeros(x.shape[1])
        m = len(x)
        beta = alpha / m

        converged = False
        while not converged:
            z = np.dot(x, th)
            g = 1
            g /= (1 + np.power(np.e, -z))
            g -= y
            temp = th - beta * np.dot(x.T, g)
            temp[1:] += l * beta * th[1:]
            diff = abs(temp - th) <= tol
            converged = np.all(diff)

            th = temp

        self._model = th

        return self

    def fit_model(self, x, y, alpha, iterations, l):
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
                :return self:  returns an instance of self.

        """
        x = self.reshape_training_set(x)

        th = np.zeros(x.shape[1])
        m = len(x)
        beta = alpha / m
        i = 0

        j_history = np.zeros(iterations)

        while i < iterations:
            z = np.dot(x, th)
            g = 1
            g /= (1 + np.power(np.e, -z))
            g -= y
            th[1:] = (1 - l * beta) * th[1:]
            th -= beta * np.dot(x.T, g)
            j_history[i] = self.compute_cost(x, y, th)
            i += 1

        self._j_history = j_history
        self._model = th

        return self

    def predict(self, x):
        """
        Predict using logistic regression

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                  Samples.
            :return: target: array, shape = (n_samples,)
                  Returns predicted values in the form of boolean values.

        """
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
        else:
            xn = np.ones(x.shape[0] + 1)
            xn[1:] = x
        return np.dot(xn, th) >= 0

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def j_history(self):
        return self._j_history

    @j_history.setter
    def j_history(self, value):
        self._j_history = value

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
    def compute_cost(X, y, theta, l):
        """
        Compute the Cost J given theta and training data
            :param X: numpy array or sparse matrix of shape [n_samples, n_features]
                Training data
            :param y: numpy array of shape (n_samples,)
                Target values
            :param theta: numpy array of shape(n_samples,)
                Linear Model
            :param l: float, optional - default is zero
                    Lambda value for Regularization term

            :return: j: float
                value of cost function for logistic regression
        """
        z = np.dot(X, theta)
        h = 1 / (1 + np.power(np.e, -z))
        m = len(X)

        ones = np.ones(y.shape)
        j = (np.dot(-y.T, np.log(h)) - np.dot((ones - y).T, np.log(ones - h))) / m
        j += l / (2.0 * m) * np.dot(theta.T, theta)

        return j
