import numpy as np
from numpy.linalg import inv
from normalization import *


class NormalEquation:

    def __init__(self, feature_scaling="auto"):
        self._model = None
        if feature_scaling == "mean":
            self.fs = MeanNormalization()
        elif feature_scaling == "std":
            self.fs = StdDeviationNormalization()
        else:
            self.fs = MeanNormalization()

    def fit(self, x, y, scale_feature=True):
        self.fs.flush()
        if scale_feature:
            x = self.fs.scale(x)

        xn = np.ones((x.shape[0], x.shape[1] + 1))
        xn[:, 1:] = x
        xtx = np.dot(xn.T, xn)
        inverse = inv(xtx)
        self._model = np.dot(inverse, np.dot(xn.T, y))

    def predict(self, x, scale_feature=True):
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
