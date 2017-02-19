from numpy import genfromtxt, concatenate, mean, unique, arange, zeros, savetxt
from numpy.random import rand, shuffle
from chart import NNChart
from classification import FastNeuralNetwork
from normalization import MeanNormalization

nnc = NNChart()
x = genfromtxt('data/train.csv', dtype=int, delimiter=',', skip_header=1)
#print test.shape

shuffle(x)
y = x[:, 0]
classes = unique(y)
x = x[:, 1:]
x = MeanNormalization().scale(x)
num = rand(64) * len(x)
#training = x[num.astype(int)]
#nnc.display(training, order='C')


fnn = FastNeuralNetwork()

hidden_units = 100
def cross_validate():
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

fnn.fit(x, y, hidden_units, l=1, maxiter=500)
test = genfromtxt('data/test.csv', dtype=int, delimiter=',',  skip_header=1)
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

print "Kaggle Accuracy detected 96.629%"
