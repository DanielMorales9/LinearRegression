from classification import FastLogisticRegression
from chart import ClassificationChart as cChart
from utility import FeatureAugmentation as FA
import numpy as np


def create_test(clf, degree, _min, step, _max):
    # create test data
    x1 = np.linspace(_min, step, _max)
    x2 = np.linspace(_min, step, _max)
    X = np.zeros((len(x1), 3))
    X[:, 0] = x1
    X[:, 1] = x2
    len1 = len(x1)
    z = np.zeros(shape=(len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([[X[i, 0], X[j, 1]]])
            t = FA.map(x, degree)
            o = np.ones(t.shape[1]+1)
            o[1:] = t.flatten()
            z[i, j] = np.dot(o, clf.model)
    return X, z

data = np.loadtxt('/Users/Daniel/PycharmProjects/MLSuite/demo/classification'
                  '/fast_logistic_regression_demo/ex2data2.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]
chart = cChart(X, y)

X = FA.map(X, 2)
clf = FastLogisticRegression()
clf.fit(X, y, l=1)
test, z = create_test(clf, 2, -1, 1.5, 50)

chart.create_visualization3D(clf, test, z)
chart.show()
