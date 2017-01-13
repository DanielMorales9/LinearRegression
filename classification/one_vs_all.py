import numpy as np
from logistic_regression import LogisticRegression


class MultiClassOneVsAll(LogisticRegression):
    """
        MultiClass Classification is based on Logistic Regression
        In practice since y = {0,1...n}, we divide our problem into
        n+1 binary classification problems; in each one, we predict
        the probability that 'y' is a member of one of our classes.
        predicts.
        We are basically choosing one class and then lumping
        all the others into a single second class.
        We do this repeatedly, applying binary logistic regression to each case,
        and then use the hypothesis that returned the highest value
        as our prediction.
    """

    def __init__(self):
        super(MultiClassOneVsAll, self).__init__()
        self._model = np.array([])
        self.y = None

    def fit(self, x, y, alpha=0.1, tol=0):
        """
        Fits a regression model (Theta) on training data

            :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                Training
            :param y: numpy array of shape (n_samples,)
                Target values
            :param alpha: float
                Learning rate
            :param tol: float
                Tollerance of convergence value
            :return self:
                Returns an instance of self.

        """
        self.y = y
        prev = None
        classes = np.unique(y)
        for c in classes:
            y_new = np.zeros(y.shape, dtype=int)
            labels = [c == e for e in y]
            y_new[np.array(labels)] = 1
            lr = super(MultiClassOneVsAll, self).fit(x, y_new, alpha, tol)
            curr = np.array([lr.model])

            if prev is None:
                prev = curr
            else:
                prev = np.concatenate((prev, curr), axis=0)

        self.model = prev
        return self

    def fit_model(self, x, y, alpha, iterations):
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
              :return self:  returns an instance of self.

        """
        self.y = y
        prev = None
        prev2 = None
        classes = np.unique(y)
        for c in classes:
            y_new = np.zeros(y.shape)
            labels = [c == e for e in y]
            y_new[np.array(labels)] = 1
            lr = super(MultiClassOneVsAll, self).fit_model(x, y_new, alpha, iterations)
            curr = np.array([lr.j_history])
            curr2 = np.array([lr.model])
            if prev is None:
                prev = curr
                prev2 = curr2
            else:
                prev = np.concatenate((prev, curr), axis=0)
                prev2 = np.concatenate((prev2, curr2), axis=0)

        self._j_history = prev[np.argmax(np.std(prev))]
        self.model = prev2
        return self

    def predict(self, x):
        """
        Predict using one vs all strategy

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                  Samples.
            :return: target: array, shape = (n_samples,)
                  Returns predicted values

        """
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = np.ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
            a = np.dot(xn, th)
            ax = 1

        else:
            xn = np.ones(x.shape[0]+1)
            xn[1:] = x
            a = np.dot(th, xn)
            ax = 0

        i = np.argmax(a, axis=ax)
        classes = np.unique(self.y)
        return classes[i]