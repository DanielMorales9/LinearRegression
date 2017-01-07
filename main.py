import numpy as np
import matplotlib.pyplot as plt
from multivariate_linear_regression import MultivariateLinearRegression
from linear_regression import LinearRegression
from normal_equation import NormalEquation


def plot_prediction(x, y, line):
    plt.plot(x, y, 'ro')
    plt.plot(line, linewidth=2.0)
    plt.axis([0, 10, -6000, -600])
    plt.show()


def plot_graph(j, i, xmin, xmax, ymin, ymax):
    plt.plot(j, i, linewidth=2)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.show()


x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])
# data init
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])
line = np.arange(0, 10000)
clf = LinearRegression()

clf.fit(x, y, alpha=0.01)
line = clf.predict(line)
#print clf.get_diagnostics()
#plot_prediction(x, y, line)


# ---> Multivariate Linear Regression Demonstration <---
print "\n---> Multivariate Linear Regression Demonstration <---\n"

data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
target = np.array([1, 2, 3])
clf = MultivariateLinearRegression()
clf.fit(data, target, alpha=0.022)
test = np.array([4, 8, 12])
print "This example ({}, {}, {}) " \
      "has been predicted as {}."\
    .format(test[0], test[1], test[2], clf.predict(test))
# I'm going to plot value of J(theta) @ iteration
#print clf.get_diagnostics()
#j = np.array(clf.j_value)

#ymin, ymax = min(j[:, 0]), max(j[:, 0])
#xmin, xmax = min(j[:, 0]), max(j[:, 1])

#xmax = 20

#plot_graph(j[:, 1], j[:, 0], xmin, xmax, ymin, ymax)


# ---> Normal Equation Demonstration <---
print "\n\n---> Normal Equation Demonstration <---\n"

data = np.array([[2104, 1416, 1534, 852],
                 [5, 3, 3, 2],
                 [1, 2, 2, 1],
                 [45, 40, 30, 36]])
data = data.T
target = np.array([460, 232, 315, 178])
clf = NormalEquation()
clf.fit(data, target)
test = np.array([800, 2, 1, 20])
print "The House with size {} (feet)," \
      " {} bedrooms, {} floor," \
      " age of home {} (years)" \
      " will cost: {} ($1000)"\
    .format(test[0], test[1], test[2], test[3], clf.predict(test))
