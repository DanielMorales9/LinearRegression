from classification import FastNeuralNetwork
import os
import scipy.io as scio
from random import random
from numpy import mean, zeros
from numpy.random import shuffle

path = os.path.dirname(os.path.abspath(__file__))
data = scio.loadmat(path+"/data/data1.mat")
x = data['X']
y = data['y']
y -= 1
y %= 10
split = random() * len(x)

X = zeros((x.shape[0], x.shape[1]+1))

X[:, :x.shape[1]] = x
X[:, -1] = y.T

shuffle(X)
x = X[:, :x.shape[1]]
y = x[:, -1]

hidden_units = 25
fnn = FastNeuralNetwork()
fnn.fit(x[:split, :], y[:split], hidden_units)
predictions = fnn.predict(x[split:, :])
print "Accuracy: {}%".format(mean(predictions == y[split:]) * 100);
