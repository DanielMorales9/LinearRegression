import numpy as np
from graph.visualize import Chart

x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])
# data init
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])

plt = Chart(x.T, y)
plt.create_convergence(0.3, iterations=100)
#plt.create_surface(0, 1)
#plt.create_contour(0, 1)
plt.show()
