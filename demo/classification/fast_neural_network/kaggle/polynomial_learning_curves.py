from numpy import array, append, max, arange
from demo.classification.fast_neural_network.kaggle.store import getStoreInstance
from classification.fast_neural_network import FastNeuralNetwork as FNN
import matplotlib.pyplot as plt
from utility.feature_augmentation import FeatureAugmentation as FA
from normalization import MeanNormalization

x, y = getStoreInstance().loadDataset('data/train.csv', isShuffle=True)
m = len(x)
idx_train = int(m * 0.60)
idx_test = int(m * 0.80)

x = FA.map2(x, degree=3)
x = MeanNormalization().scale(x)
x_train = x[:idx_train, :]
y_train = y[:idx_train]
x_val = x[idx_train:idx_test, :]
y_val = y[idx_train:idx_test]
x_test = x[idx_test:, :]
y_test = y[idx_test:]

fnn = FNN()
error_train = array([])
error_val = array([])
examples = range(2100, idx_train+1, 2100)

for i in examples:
    fnn.fit(x_train[:i, :], y[:i], 100)
    j_train = fnn.compute_cost(x_train[:i, :], y[:i])
    error_train = append(error_train, j_train)
    j_val = fnn.compute_cost(x_val, y_val)
    error_val = append(error_val, j_val)

plt.plot(range(2100, idx_train+1, 2100), error_train, linewidth=2)
plt.plot(range(2100, idx_train+1, 2100), error_val, linewidth=2)
legend = ["Error Train", "Error CV"]
plt.ylabel("Error")
plt.xlabel("Number of examples (m)")
plt.legend(legend)
plt.title("Learning curve for neural network.")
plt.show()

#It suffers of high variance
