import numpy as np
import matplotlib.pyplot as plt

# data init
x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])


def simple_linear_regression(x, y, alpha=0.1):
    bound = 0
    th0 = 0.0
    th1 = 0.0
    m = len(x)
    beta = alpha / m
    iter = 0
    converged = False
    while not converged:
        error = (th0 + th1 * x) - y
        temp0 = th0 - np.sum(error) * beta
        temp1 = th1 - np.sum(error * x) * beta
        diff0 = abs(temp0 - th0)
        diff1 = abs(temp1 - th1)
        converged = diff0 <= bound and diff1 <= bound
        th0 = temp0
        th1 = temp1
        iter += 1
    return th0, th1, iter

th0, th1, iteration = simple_linear_regression(x, y, alpha=0.01)
print "Theta0: {}, Theta1: {}".format(th0, th1)
print "Number of iteration {}".format(iteration)

plt.plot(x, y, 'ro')
line = np.arange(0, 10000)
line = th0 + th1*line
plt.plot(line, linewidth=2.0)
plt.axis([0, 10, -6000, -600])
plt.show()
