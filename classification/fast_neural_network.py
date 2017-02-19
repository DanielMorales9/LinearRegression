from classification import LogisticRegression
from numpy import shape, dot, e, ones, log, sum, unique, \
    power, zeros, reshape, concatenate, argmax
from numpy.random import rand
from random import random
from scipy.optimize import fmin_l_bfgs_b

class FastNeuralNetwork(LogisticRegression):

    def __init__(self):
        super(FastNeuralNetwork, self).__init__()
        self.l = None
        self.X = None
        self.y = None
        self.il = None
        self.hl = None
        self.nl = None

    def fit(self, x, y, hidden_units, l=0, maxiter=50):
        """
            Fits a model using CG Optimization method.

                :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                    Training data
                :param y: numpy array of shape (n_samples,)
                    Target values
                :param hidden_units: int
                    Number of hidden units in hidden layers
                :param l: float, optional - default is zero
                    Lambda value for Regularization term
                :param maxiter: int, optional - default is 50
                    Number of maxiter for CG optimization method
        """
        self.l = l
        self.hl = hidden_units
        self.y = self.reshape_labels(y)
        _, self.il = shape(x)
        _, self.nl = shape(self.y)
        th1, th2 = self.initWeight(self.il, self.hl, self.nl)
        self.X = self.reshape_training_set(x)
        g1 = th1.flatten(order='F')
        g2 = th2.flatten(order='F')
        init_theta = concatenate((g1, g2))

        self._model, _, _= fmin_l_bfgs_b(self.cost_function, init_theta, maxiter=maxiter)

    def forward_propagation(self, theta, il, hl, nl, X):
        """
        Neural Network Forward Propagation Algorithm
        :param theta: numpy array
            Weights of the Neural Network
        :param il: int
            Number of input layer units
        :param hl: int
            Number of hidden layer units
        :param nl: int
            Number of output layer units
        :param X: array like or sparse matrix
            Training set
        :return: tuple
            th1: first weights matrix
            th2: second weights matrix
            z2: output of first propagation
            aa2: sigmoid on z2 + bias term
            a3: output of forward propagation
        """
        shp = hl * (il + 1)
        th1 = reshape(theta[0:shp], (hl, (il+1)), order='F')
        th2 = reshape(theta[shp:], (nl, (hl+1)), order='F')
        z2 = dot(X, th1.T)
        a2 = self.sigmoid(z2)
        aa2 = ones((a2.shape[0], a2.shape[1] + 1))
        aa2[:, 1:] = a2
        a3 = self.sigmoid(dot(aa2, th2.T))
        return th1, th2, z2, aa2, a3

    def cost(self, a3, y, th1, th2, l):
        """
        Compute cost
        :param a3: numpy array
            output of Forward Propagation
        :param y:   numpy matrix
            Label set
        :param th1: numpy array
            first weights matrix
        :param th2: numpy array
            second weights matrix
        :param l: float,
            Lambda for Regulazation Term
        :return: float
            cost value
        """
        j = 0
        for ai, yi in zip(a3, y):
            j -= dot(log(ai), yi.T) + dot(log(1 - ai), (1 - yi).T)
        m = len(y)
        j /= m
        beta = l / (2 * m)
        j += beta * (sum(power(th1[:, 1:], 2)) + sum(power(th2[:, 1:], 2)))
        return j

    def initWeight(self, f, h, k):
        """
        Weight Initialization
        :param f: int
            Number of input layer
        :param h: int
            Number of hidden layer
        :param k: int
            Number of output layer
        :return: tuple
            th1: first weights matrix
            th2: second weights matrix
        """
        init_epsilon = 0.12
        th1 = rand(h, 1 + f) * 2 * init_epsilon - init_epsilon
        th2 = rand(k, 1 + h) * 2 * init_epsilon - init_epsilon
        return th1, th2

    def gradient(self, X, y, l, th1, th2, a3, aa2, z2):
        """
        Computes the gradient function
        :param X: matrix
            Training set
        :param y: matrix
            Label set
        :param l: float
            Lambda Regularization Term
        :param th1: numpy array
            First weights matrix
        :param th2: numpy array
            Second weights matrix
        :param a3:  numpy array
            Output of Forward propagation
        :param aa2: numpy array
            Sigmoid on z2 + bias term
        :param z2: numpy array
            Output of first propagation without sigmoid
        :return: numpy array
            Gradient

        """
        m = len(X)
        d3 = a3 - y
        z = self.sigmoid(z2) * (1 - self.sigmoid(z2))
        d2 = dot(d3, th2[:, 1:]) * z
        th1_grad = dot(d2.T, X)
        th2_grad = dot(d3.T, aa2)

        th1_grad /= m
        th2_grad /= m
        beta = l / m
        th1_grad[:, 1:] += th1[:, 1:] * beta
        th2_grad[:, 1:] += th2[:, 1:] * beta

        g1 = th1_grad.flatten(order='F')
        g2 = th2_grad.flatten(order='F')
        grad = concatenate((g1, g2))

        return grad

    def sigmoid(self, z):
        return 1.0 / (1.0 + e**(-z))

    def cost_function(self, theta):
        """
            Cost function computes the model's cost and gradient
            :param theta: numpy array
                Linear Model
            :return: tuple: float, numpy array
                cost and gradient array
        """
        th1, th2, z2, aa2, a3 = \
            self.forward_propagation(theta, self.il, self.hl, self.nl, self.X)
        j = self.cost(a3, self.y, th1, th2, self.l)
        grad = self.gradient(self.X, self.y, self.l, th1, th2, a3, aa2, z2)
        return j, grad

    def predict(self, x):
        """
        Predict using Neural Network

            :param x: {array-like, sparse matrix}, shape = (n_samples, n_features)
                  Samples.
            :return: target: array, shape = (n_samples,)
                  Returns predicted values

        """
        th = self._model
        # checks whether the example to predict is a vector or a matrix
        if len(x.shape) > 1:
            xn = ones((x.shape[0], x.shape[1] + 1))
            xn[:, 1:] = x
            ax = 1

        else:
            xn = ones(x.shape[0] + 1)
            xn[1:] = x
            ax = 0

        a = self.forward_propagation(self._model, self.il, self.hl, self.nl, xn)[4]

        return argmax(a, axis=ax)

    @staticmethod
    def reshape_labels(y):
        """
        Reshape Label list into a 0, 1 matrix

        :param y: array_like (nsample,)
            y must have classes from 0 to k-1
        :return: numpy matrix
        """
        classes = unique(y)
        yy = zeros((len(y), classes[-1]+1))
        for i in range(len(y)):
            yy[i, y[i]] = 1
        return yy

    @staticmethod
    def compute_cost(X, y, theta, l):
        """
            Compute the Cost J given theta and training data
                :param X: numpy array or sparse matrix of shape [n_samples, n_features]
                    Training data
                :param y: numpy array of shape (n_samples,)
                    Target values
                :param theta: numpy array
                    Neural Network model
                :param l: float, optional - default is zero
                    Lambda value for Regularization term

                :return: j: float
                    value of cost function for logistic regression
        """
        nn = FastNeuralNetwork()
        nn.X = FastNeuralNetwork.reshape_training_set(X)
        nn.y = FastNeuralNetwork.reshape_labels(y)
        j, _ = nn.cost_function(theta)
        return j