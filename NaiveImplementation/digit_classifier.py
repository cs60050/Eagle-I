#additional libraries
import numpy as np
import theano 
from theano import tensor as T
from theano import function, shared, In
from matplotlib import pyplot as plt
import pandas as pd
import time


#random number generator
rng = np.random

#details regarding training set
#number of training examples
N = 5000
#number of features
features = 784
#number of output classes
out_class = 10
#learning rate
alpha = 0.00001

training_steps = 1000

#read the data
print 'Reading data...'
train = pd.read_csv('./../Data/train.csv')
print 'Done with reading data...'

#split data into input and output values
X_train = train.iloc[0:5000, 1:]
Y_train = train['label']
Y_train = Y_train[0:5000]

#declare variables
# x- (1x785) and y- (1x10)
# 'x' is matrix of 'Nxfeatures' values 
x = T.dmatrix('x')
# 'y' is vector of 10 values (output) corresponding to each digit
y = T.ivector('y')

#declaring weights and bias term for all the classes
W = np.zeros((features, out_class)).astype(theano.config.floatX)
B = np.zeros((1, out_class)).astype(theano.config.floatX)
w = shared(np.zeros((features), dtype=theano.config.floatX), name='w')
b = shared(0.0, name='b')

#hypothesis function is basically thus
h = 1 / (1 + T.exp(-T.dot(x, w) - b))
#predict true if hypothesis is greater than 0.5
H = 1 / (1 + T.exp(-T.dot(x, W) - B))

#had to change this from theano.tensor to numpy?
pred = T.argmax(H, axis=1)
#for i in xrange(N):
#	pred[i] = T.argmax(H[i])


#cost function
J = -y * T.log(h) - (1-y) * T.log(1-h)
cost = J.mean()
#add reguralization term to the cost function
reg = 0.01 * (w ** 2).sum()
cost = cost + reg
#calculate the gradients of parameters
grad_w, grad_b = T.grad(cost, [w, b])

#function for training and predicting
train = function(inputs=[x, y], outputs=[pred, cost],\
                 updates=[(w, w-alpha * grad_w), (b, b-alpha*grad_b)])

predict = function(inputs=[x], outputs=pred)


#training the model separately for each class
for i in xrange(out_class):
	Y_vec_train = np.zeros((N,))
	Y_dummy = Y_train == i

	#create vector output consisting of zeros and ones
	for k in xrange(N):
		if Y_dummy[k] == True:
			Y_vec_train[k] = 1
	
	#declare the weights and bias term 
	w = shared(rng.randn((features)), name='w')
	#why didn't bias term update when initialised with 'zero'
	b = shared(rng.randn(), name='b')

	print '\n\nTraining the parameters for class ', i
	print 'Iterating over entire training set\n'
	#train the parameters for this particular class
	cost_vec = np.zeros((training_steps, ))
	for j in xrange(training_steps):
		pred, cost = train(X_train, Y_vec_train)
		print 'Iteration: ', j+1, '\tCost: ', cost
		cost_vec[j] = cost

	#plot cost as a function of weights 
	x_vals = [idx for idx in range(training_steps)]
	y_vals = cost_vec
	plt.plot(x_vals, y_vals, 'r')
	plt.savefig("./cost{i}.png".format(i=i))
	plt.show()
	plt.cla()
	plt.clf()

	#store the weights of parameters for that particular class
	print 'Done with training for class ', i

   	print 'weights for class: ', i, w.get_value()
   	print 'bias term for class: ', i, b.get_value()

	W[:, i] = w.get_value()
	B[0, i] = b.get_value()	

	#debug here...are these values stored correctly??
	#print 'Weights are as follows: \n'
	#print 'Unbiased weights: \n', W[:, i]
	#print '\nBiased weight: \n', B[0, i]
	#print 'Stored the weights for class ', i
	time.sleep(2)
	
print 'Done with training the model!'

#save parameters for later use 
#ANALYSIS DONE...SAVED CORRECTLY
np.savez('./weights.npz', w=W)
np.savez('./bias.npz', b=B)
print 'Saved the parameters...'
