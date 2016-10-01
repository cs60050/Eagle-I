import numpy as np
import pandas as pd
import theano

#read weights and bias term
weights = np.load('./weights.npz')
bias = np.load('./bias.npz')
weights = weights['w']
bias = bias['b']
train = pd.read_csv('./../Data/train.csv')
X = train.iloc[0:5000, 1:]
Y = train['label']
Y = Y[0:5000]

X = X.as_matrix()
X = X.astype(theano.config.floatX)

#hypothesis function
h = 1 / (1 + np.exp(-np.dot(X, weights) - bias))
count = 0
N = 5000
pred = np.zeros((N, 1))
print h[0:10]

print 'Shape of hypothesis over training set: ', h.shape
for i in xrange(N):
	pred[i] = np.argmax(h[i])

for i in xrange(N):
	if pred[i] == Y[i]:
		count += 1

#print accuracy over the given set of examples
print 'Accuracy : ', (count/42000.0) * 100
print 'Done!!'
