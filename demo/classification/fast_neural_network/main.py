from classification import FastNeuralNetwork
import os
import scipy.io as scio
from random import random
from numpy import mean, zeros, reshape, ceil
from numpy.random import shuffle, rand
from chart import NNChart

nnc = NNChart()

path = os.path.dirname(os.path.abspath(__file__))
data = scio.loadmat(path+"/data/data1.mat")
x = data['X']
y = data['y']
y -= 1
y %= 10

num = rand(64) * len(x)
training = x[num.astype(int)]
nnc.display(training)

split = int(round(random() * len(x)))
X = zeros((x.shape[0], x.shape[1]+1))

X[:, :x.shape[1]] = x
X[:, -1] = y.T

shuffle(X)
x = X[:, :x.shape[1]]
y = x[:, -1]
y = y.astype(int)
hidden_units = 25
fnn = FastNeuralNetwork()
fnn.fit(x[:split, :], y[:split], hidden_units)
predictions = fnn.predict(x[split:, :])

theta = fnn.model
print "Accuracy: {}%".format(mean(predictions == y[split:]) * 100)
th1 = reshape(theta[:fnn.hl * (fnn.il + 1)], (fnn.hl, (fnn.il + 1)), order='F')

nnc.display(th1[:9, 1:])
