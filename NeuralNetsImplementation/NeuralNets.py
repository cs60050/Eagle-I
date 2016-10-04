import time
import numpy as np
from matplotlib import pyplot as plt  
import pandas as pd
import theano
from theano import function, In, shared
from theano import tensor as T

rng = np.random

#read data
print 'Reading data...'
train = pd.read_csv('./../Data/train.csv')
test = pd.read_csv('./../Data/test.csv')

#separate data
X_train = train.iloc[0:, 1:]
Y_train = train['label']
X_test = test.iloc[:, :]

#additional details of data set
M = X_train.shape[0]
learning_rate = 0.01
training_steps = 100

#convert data into numpy arrays
X_train = X_train.as_matrix()
Y_train = Y_train.as_matrix()


#network architecture details
L = 3
input_units = 784
hidden_units = 625
output_units = 10


#convert Y_train into one hot vectors
print 'creating one hot vectors...'
Y_vec_train = np.zeros((M, output_units))
for i in xrange(output_units):
	Y_dummy = (Y_train == i)

	#create hot vector
	for k in xrange(M):
		if Y_dummy[k] == True:
			Y_vec_train[k][i] = 1

#do feature scaling over data
max_pixel_val = 255
min_pixel_val = 0
X_train = X_train / (max_pixel_val - min_pixel_val)
X_test = X_test / (max_pixel_val - min_pixel_val)

#symbolic variables
x = T.dmatrix('x')
y = T.dmatrix('y')


#initialise random weights
print 'Initialising weights...'
W1_vals = np.asarray(rng.randn(input_units, hidden_units), dtype=theano.config.floatX)
W1 = shared(value=W1_vals, name='W1')
b1 = shared(value=rng.randn(hidden_units, ), name='b1')

W2_vals = np.asarray(rng.randn(hidden_units, output_units), dtype=theano.config.floatX)
W2 = shared(value=W2_vals, name='W2')
b2 = shared(value=rng.randn(output_units, ), name='b2')


#feed forward activations
hidden_activations = T.nnet.sigmoid(T.dot(x, W1))
prob_y_given_x = T.nnet.sigmoid(T.dot(hidden_activations, W2))
predicted_idx = T.argmax(prob_y_given_x, axis=1)

#cost 
cost = T.mean(T.nnet.categorical_crossentropy(prob_y_given_x, y))
params = [W1, W2]
gradients = T.grad(cost, params)
updates = [(param, param - learning_rate * grad) for param, grad in zip(params, gradients)]


#compile functions
print 'Compiling functions to train and predict...'
train = function(inputs=[x, y], outputs=cost, updates=updates)
predict = function(inputs=[x], outputs=[prob_y_given_x, predicted_idx])


#train the model
for i in range(training_steps):
	_cost = train(X_train, Y_vec_train)
	print 'Iteration: ', i+1, ' Cost: ', _cost





