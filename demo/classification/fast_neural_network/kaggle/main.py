from numpy import genfromtxt, concatenate, mean, \
    unique, arange, zeros, savetxt, array, append, argmin
from numpy.random import rand, shuffle
from classification import FastNeuralNetwork as FNN
from normalization import MeanNormalization
from utility.feature_augmentation import FeatureAugmentation as FA
from chart import NNChart

x = genfromtxt('data/train.csv', dtype=int, delimiter=',', skip_header=1)

shuffle(x)
y = x[:, 0]
classes = unique(y)
x = x[:, 1:].astype(float)
num = rand(64) * len(x)
training = x[num.astype(int)]
NNChart().display(training, order='C')
x = MeanNormalization().scale(x)

"""
def cross_validate():
    fnn = FNN()
    k = 5
    start = 0
    stop = len(x) / k
    meany = 0.0
    for i in range(k):
        fnn.fit(x[start:stop, :], y[start:stop], hidden_units)
        if start != 0:
            kfold = concatenate((x[0:start, :], x[stop:, :]), axis=0)
            yfold = concatenate((y[0:start], y[stop:]))
        else:
            kfold = x[stop:, :]
            yfold = y[stop:]
        predictions = fnn.predict(kfold)
        meany += mean(predictions == yfold)

    meany /= float(k)

    print meany * 100
"""

fnn = FNN()
fnn.fit(x, y, 128, l=163, maxiter=50)
test = genfromtxt('data/test.csv', dtype=int, delimiter=',',  skip_header=1)
#test = FA.map2(test, degree=3)
test = MeanNormalization().scale(test)
prediction = fnn.predict(test)
plen = len(prediction)

id = arange(1, plen+1, 1)

mat = zeros((plen, 2))

mat[:, 0] = id
mat[:, 1] = prediction

savetxt('data/submission.csv',
            mat,
           delimiter=',',
            fmt='%d',
           newline='\n',
           header='ImageId,Label')

#print "Kaggle Accuracy detected 96.629%"