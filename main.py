import numpy as np
from graph.visualize import LinearRegressionChart
from classification import LogisticRegression

x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])

l = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# data init
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])

clf = LogisticRegression()

print clf.fit(x, l).predict(np.array([1]))


plt = LinearRegressionChart(x.T, y)
plt.create_convergence(0.01, 1, iterations=100)
plt.create_surface(0, 1)
plt.create_contour(0, 1)
plt.show()
