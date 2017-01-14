import numpy as np
from abc import ABCMeta, abstractmethod


class Normalization:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.feature_range = None
        self.feature_mean = None

    @abstractmethod
    def scale(self, x): pass

    def flush(self):
        self.feature_range = None
        self.feature_mean = None


class MeanNormalization(Normalization):
    """
    Mean Normalization algorithm
    """

    def __init__(self):
        super(MeanNormalization, self).__init__()

    def scale(self, x):
        """
        Scales the training set by subtracting the mean value and dividing by the range of value of each feature
        :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Training set.
        :return: xs: {array-like, sparse matrix}, shape = (n_samples, n_features)
            Scaled Training set.
        """

        if self.feature_mean is None:
            self.feature_mean = np.mean(x, axis=0)
        if self.feature_range is None:

            feature_max = np.amax(x, axis=0)
            feature_min = np.amin(x, axis=0)
            self.feature_range = feature_max - feature_min

            # this will prevent division by zero for features with no range
            if len(x.shape) > 1:
                self.feature_range[feature_max == feature_min] = 1
            elif feature_max == feature_min:
                self.feature_range = 1
        xn = x - self.feature_mean
        xn /= self.feature_range

        return xn


class ZScoreNormalization(Normalization):
    """
    ZScore Normalization Algorithm
    """

    def __init__(self):
        super(ZScoreNormalization, self).__init__()

    def scale(self, x):
        """
        Scales the training set by subtracting the mean value and dividing by the standard deviation
            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                   Training set.
            :return: xs: {array-like, sparse matrix}, shape = (n_samples, n_features)
                   Scaled Training set.
        """
        if self.feature_mean is None:
            feature_mean = np.mean(x, axis=0)

        if self.feature_range is None:
            self.feature_range = np.std(x, axis=0)

            # this will prevent division by zero for features with no std
            if len(x.shape) > 1:
                self.feature_range[self.feature_range == 0] = 1
            elif self.feature_range == 0:
                self.feature_range = 1

        xn = x - self.feature_mean
        xn /= self.feature_range

        return xn
