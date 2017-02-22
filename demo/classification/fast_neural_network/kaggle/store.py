from numpy import genfromtxt, array
from numpy.random import shuffle

class Store:

    def __init__(self):
        self.x = None
        self.y = None
        self.test = None

    def loadDataset(self, path, delim=',', skip=1, isShuffle=False):
        if self.x is  None and self.y is None:
            x = genfromtxt(path, dtype=int, delimiter=delim, skip_header=skip)
            if isShuffle:
                shuffle(x)
            self.y = x[:, 0]
            self.x = x[:, 1:].astype(float)
        return self.x, self.y

    def loadTest(self, path, delim=',', skip=1):
        if self.test is None:
            self.test = \
                genfromtxt(path, dtype=int, delimiter=delim, skip_header=skip)
        return self.test

    def delete(self):
        self.x = None
        self.y = None
        self.test = None

this = Store()

def getStoreInstance():
    return this