import numpy as np
from graph.visualize import Chart
from regression import *

x = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10])

y = np.array([0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
y = np.array([-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157])

clf1 = MultivariateLinearRegression()
clf = FastMultivariateLinearRegression()
print "Fast fitting"
print clf.fit(x, y, l=0.1)

print "Fast Prediction"
print clf.predict(np.array([1]))

X = clf1.reshape_training_set(x)
print "Normal fitting"
print clf1.fit(x, y, l=0.1).model, clf1.compute_cost(X, y, clf1.model, l=0.1)
print "Normal Prediction"
print clf1.predict(np.array([1]))

X = np.array([[e] for e in x])
plt = Chart(X, y)
test = np.array([[e] for e in range(0, 10)])
plt.create_visualization(clf, test)

plt.create_visualization(clf1, test)
plt.show()