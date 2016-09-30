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
N = 42000
#number of features
features = 784
#number of output classes
out_class = 10
alpha = 0.01
training_steps = 1000

#read the data
train = pd.read_csv('./../Data/train.csv')

#split data into input and output values
X_train = train.iloc[:, 1:]
Y_train = train['label']

#declare variables
# x- (1x785) and y- (1x10)
# 'x' is matrix of 'Nxfeatures' values 
x = T.dmatrix('x')
# 'y' is vector of 10 values (output) corresponding to each digit
y = T.dvector('y')

#declaring weights and bias term for all the classes
W = np.zeros((features, out_class)).astype(np.float64)
B = np.zeros((1, out_class)).astype(np.float64)
w = shared(rng.randn(features), name='w')
b = shared(rng.randn(), name='b')

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
  	w = shared(rng.randn(features), name='w')
	#why didn't bias term update when initialised with 'zero'
	b = shared(rng.randn(), name='b')

	print '\n\nTraining the parameters for class ', i
	print 'Iterating over entire training set\n'
	#train the parameters for this particular class
	cost_vec = np.zeros((N, ))
	for j in xrange(training_steps):
		pred, cost = train(X_train, Y_vec_train)
		print 'Iteration: ', j+1, '\tCost: ', cost
		cost_vec[j] = cost

	#plot cost as a function of weights 
	x_vals = [i for i in range(training_steps)]
	y_vals = cost_vec
	plt.plot(x_vals, y_vals, 'r')
	plt.savefig('./cost.png')
	plt.show()
	
	#store the weights of parameters for that particular class
	print 'Done with training for class ', i
   
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

#print 'Entire weight set: \n'
#print W
#print 'Vector of biased terms for 10 classes are: \n'
#B


#predict the accuracy over the training set 
predicted = predict(X_train)

#predict seems not to be working...
#this should be of dimension Nxout_class
#this is probability distribution
np.savez('./predicted.npz', pred=predicted)


print 'Comparing target value and predicted value for training set:\n'
#for tar, pre in zip(Y_train, predicted):
#	print 'Target: ', tar, 'Predicted: ', pre
#print predicted



