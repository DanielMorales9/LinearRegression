import numpy as np
import matplotlib.pyplot as plt
from multivariate_linear_regression import MultivariateLinearRegression
from linear_regression import LinearRegression


def plot_prediction(x, y, line):
    plt.plot(x, y, 'ro')
    plt.plot(line, linewidth=2.0)
    plt.axis([0, 10, -6000, -600])
    plt.show()

# data init
x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])
line = np.arange(0, 10000)

clf = LinearRegression()

clf.fit(x, y, alpha=0.01)
line = clf.predict(line)
print clf.get_diagnostics()
plot_prediction(x, y, line)

data = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
target = np.array([1, 2, 3])
clf = MultivariateLinearRegression()
clf.fit(data, target, alpha=0.01)
print clf.predict(np.array([4, 8, 12]))
