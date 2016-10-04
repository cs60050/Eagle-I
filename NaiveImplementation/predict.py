import numpy as np
import pandas as pd
import theano
from theano import function
import time
from theano import tensor as T

#read weights and bias term
weights = np.load('./weights.npz')
bias = np.load('./bias.npz')
w = weights['w']
b = bias['b']

# print 'weights shape: ', w.shape
# print 'bias shape: ', b.shape 

#constants
M = 5000

#read the data
train = pd.read_csv('./../Data/train.csv')

#split data into input and output values
X_train = train.iloc[0:5000, 1:]
Y_train = train['label']
Y_train = Y_train[0:5000]

#convert the values to numpy arrays
X_train = X_train.as_matrix()
X_train = X_train.astype(theano.config.floatX)
Y_train = Y_train.astype(int)

#do the feature scaling of parameters...belong to [0, 1]
max_pixel_val = 255
min_pixel_val = 0
X_train = X_train / (max_pixel_val - min_pixel_val)


#declare variables
x = T.dmatrix('x')

#hypothesis function
h = 1.0 / (1.0 + T.exp(-T.dot(x, w) - b))
pred_vec = T.argmax(h, axis=1)
# print 'w[:, 0] :', [idx1 for idx1 in w[:, 0]]
# print 'b[0][0]: ', b[0][0]

#predict the hypothesis only for the class i
# h_0 = 1.0 / (1.0 + T.exp(-T.dot(x, w[:, 0]) - b[0][0]))

#compile
predict = function(inputs=[x], outputs=[h, pred_vec])
# predict_0 = function(inputs=[x], outputs=h_0)

# debugging function
print 'reading hypothesis...'
#predict values
hypo, predicted = predict(X_train)
# predicted_0 = predict_0(X_train)
# print 'hypo shape: ', hypo.shape, ' predicted shape: ', predicted.shape
# print 'hypothesis: ', hypo[0:20, :]
# print 'corresponding class: ', predicted[0:20]
# print 'X_train[0, :] : \n', [idx2 for idx2 in X_train[0, :]]
# print '\npredictions for class 0: \n', predicted_0

print 'Predicting over training set: \n'
count = 0
for i in xrange(M):
	# print 'Iteration: ', i , '\tpredicted: ', predicted[i], '\ttarget: ', Y[i]
	# time.sleep(0.1)
	if predicted[i] == Y_train[i]:
		count += 1

#print accuracy over the given set of examples
print 'Accuracy : ', (count/5000.0) * 100, '%'
print 'Done!!'