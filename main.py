import numpy as np
#from graph.visualize import Chart
from classification import *

x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])

l = np.array([2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# data init
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])

clf = MultiClassOneVsAll()

clf.fit(x, l, alpha=0.31, tol=10**-4)
print clf.predict(np.array([1]))


l = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


#plt = Chart(x.T, l)
#plt.create_convergence(LogisticRegression(), 0.01, stop_alpha=1, iterations=10000)
#plt.create_surface(0, 1)
#plt.create_contour(0, 1)
#plt.show()

