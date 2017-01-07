import numpy as np
from abc import ABCMeta, abstractmethod


class Normalization:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def scale(self, x): pass

    @abstractmethod
    def flush(self): pass


class MeanNormalization(Normalization):

    def __init__(self):
        super(MeanNormalization, self).__init__()
        self.feature_range = None
        self.feature_mean = None

    def scale(self, x):
        if self.feature_mean is None:
            self.feature_mean = np.mean(x, axis=0)
        if self.feature_range is None:

            feature_max = np.amax(x, axis=0)
            feature_min = np.amin(x, axis=0)
            self.feature_range = feature_max - feature_min
            self.feature_range[feature_max == feature_min] = 1

        xn = x - self.feature_mean
        xn /= self.feature_range

        return xn

    def flush(self):
        self.feature_range = None
        self.feature_mean = None


class StdDeviationNormalization(Normalization):

    def __init__(self):
        super(StdDeviationNormalization, self).__init__()
        self.feature_range = None
        self.feature_mean = None

    def scale(self, x):
        if self.feature_mean is None:
            feature_mean = np.mean(x, axis=0)

        if self.feature_mean is None:
            self.feature_range = np.std(x, axis=0)

            if self.feature_range == 0:
                self.feature_range = 1

        xn = x - self.feature_mean
        xn /= self.feature_range

        return xn

    def flush(self):
        self.feature_range = None
        self.feature_mean = None
