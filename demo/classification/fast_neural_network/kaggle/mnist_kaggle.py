from numpy import genfromtxt, concatenate, mean, \
    unique, arange, zeros, savetxt, array, append, argmin
from numpy.random import rand, shuffle
from classification import FastNeuralNetwork as FNN
from normalization import MeanNormalization
from utility.feature_augmentation import FeatureAugmentation as FA

x = genfromtxt('data/train.csv', dtype=int, delimiter=',', skip_header=1)

shuffle(x)
y = x[:, 0]
classes = unique(y)
x = x[:, 1:].astype(float)
#x = MeanNormalization().scale(x)
#num = rand(64) * len(x)
#training = x[num.astype(int)]
#NNChart().display(training, order='C')

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

cv_index = int(len(x) * 0.60)
test_index = int(len(x) * 0.80)

backoff1 = [0.01 * (2 ** i) for i in arange(1, 11)]
backoff1.insert(0, 0.01)
backoff1.insert(0, 0)
hidden_units = 128

choosen = array([])
for i in backoff1:
    cost_values = array([])
    nn_instances = array([])
    for j in range(1, 4):
        X = FA.map2(x, degree=j)
        X = MeanNormalization().scale(X)
        fnn = FNN()
        fnn.fit(X[:cv_index, :], y[:cv_index], hidden_units, l=i, maxiter=50)
        cost = fnn.compute_cost(X[cv_index:test_index],
                                y[cv_index:test_index])
        print cost, i, j
        nn_instances = append(nn_instances, [fnn])
        cost_values = append(cost_values, cost)
    fai = argmin(cost_values)
    fa = range(1, 4)[fai]
    choosen = append(choosen, (min(cost_values), i, fa, nn_instances[fai]))
cost_values = array([cost[0] for cost in choosen])
index = argmin(cost_values)
cost, l, fa, fnn = choosen[index]

print "Feature Augmentation {}".format(fa)
print "Value of lambda {}".format(l)
print "Cost of validation {}".format(cost)

X = FA.map2(x, degree=fa)
X = MeanNormalization.scale(X)
fnn.fit(X[:cv_index, :], y[:cv_index], hidden_units, l=l, maxiter=50)
predictions = fnn.predict(X[test_index:, :])
print "Accuracy {}".format(mean(predictions == y[test_index:]) * 100)
cost = fnn.compute_cost(X[test_index:, :],
                        y[test_index:])
print "Test Cost {}".format(cost)

"""
fnn = FNN()
fnn.fit(x, y, 128, l=0.16, maxiter=50)



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
"""
