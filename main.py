from classification import FastNeuralNetwork
import numpy as np

fnn = FastNeuralNetwork(np.array([1]))
fnn.model = [np.array([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]]),
             np.array([[1, 1, 1, 1]])]

x = fnn.reshape_training_set(np.array([[1, 2, 3],
                                       [1, 2, 3],
                                       [1, 2, 3]]))
r = fnn.forward_propagation(x)
