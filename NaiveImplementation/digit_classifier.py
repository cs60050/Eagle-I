import numpy as np
import theano 
from theano import tensor as T
from theano import function, shared, In
from matplotlib import pyplot as plt
import pandas as pd
import time


#random number generator
rng = np.random
#number of training examples
M = 5000
#number of features
N = 784
#number of output classes
out_class = 10
#learning rate
alpha = 0.1
training_steps = 1000

#read the data
print 'Reading data...'
train = pd.read_csv('./../Data/train.csv')
print 'Done with reading data...'

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
y = T.dvector('y')

#declaring weights and bias term for all the classes
W = np.zeros((N, out_class))
B = np.zeros((1, out_class))
w = shared(value=rng.randn((N)), name='w')
b = shared(value=rng.randn(), name='b')

#hypothesis function is basically thus
h = 1.0 / (1.0 + T.exp(-T.dot(x, w) - b))
pred = h > 0.5

# H = 1.0 / (1.0 + T.exp(-T.dot(x, W) - B))
# pred_vec = T.argmax(H, axis=1)

#cost function with regularization
J = -y * T.log(h) - (1-y) * T.log(1-h)
reg = 0.01 * (w ** 2).sum()
cost = J.mean() + reg

#calculate the gradients of parameters
grad_w, grad_b = T.grad(cost, [w, b])

#function for training and predicting
train = function(inputs=[x, y], outputs=[pred, cost], \
	             updates=[(w, w - alpha * grad_w), (b, b - alpha*grad_b)])
# predict = function(inputs=[x], outputs=[H, pred_vec])

#new predict function for a given class DEBUGGING
predict_class = function(inputs=[x], outputs=[h, pred])

"""
TRAIN THE MODEL
"""
#training the model separately for each class
for i in xrange(out_class):

	#create a M-dimensional vector of 0s and 1s for that class
	Y_vec_train = np.zeros((M, ))
	Y_dummy = (Y_train == i)
	for k in xrange(M):
		if Y_dummy[k] == True:
			Y_vec_train[k] = 1

	# if i == 0:
	# 	print 'random b: ', b.get_value()
	# 	print 'random w: ', w.get_value()

	#set the random values to the weights and bias terms
	w.set_value(rng.randn((N)))
	b.set_value(rng.randn())

	print '\nTraining the parameters for class ', i
	# print 'Iterating over entire training set...\n'
	#train the parameters for this particular class
	cost_vec = np.zeros((training_steps, ))
	for j in xrange(training_steps):
		pred, cost = train(X_train, Y_vec_train)
		# print 'updated b: ', b.get_value()
		# print 'updated w: ', w.get_value()[0:10]
		# time.sleep(0.5)
		print '\n'
		print 'Iter: ', j+1, '\tJ: ', cost, \
		      '\tb: ', b.get_value(), '\tw: ', w.get_value()[0:3]
		cost_vec[j] = cost

	#predictions made by the model at the end of training for class i
	# print '\n\nPredicting the output after the training for class ', i
	pred_count = 0
	for predicted, target in zip(pred, Y_vec_train):
		if predicted == target:
			pred_count += 1
		# print 'predicted: ', predicted, ' Target: ', target
	print 'Accuracy in classification for class ', i, ' is: ', (pred_count/5000.0) * 100

	#plot cost as a function of weights 
	x_vals = [idx for idx in range(training_steps)]
	y_vals = cost_vec
	plt.plot(x_vals, y_vals, 'r')
	plt.savefig("./cost_for_class_{i}.png".format(i=i))
	plt.show()
	plt.cla()
	plt.clf()

	#store the weights of parameters for that particular class
	# print 'Done with training for class ', i

   	# print 'weights for class: ', i, w.get_value()
   	# print 'bias term for class: ', i, b.get_value()

   	#predict the hypothesis for the classes
   	hypo, pr = predict_class(X_train)
  #  	if i == 0:
  #  		# print 'Xtrain[0, :] : \n', [idx2 for idx2 in X_train[0, :]]

		# print hypo
		# print pr
		# # print '\ntrained b: ', b.get_value()
		# # print 'trained w: ', w.get_value()
		# # print 'weights:\n', w.get_value()
		# # print '\n\nbias: \n', b.get_value()

  #  	# print w.get_value()[0:20]
  #  	# print b.get_value()


	W[:, i] = w.get_value()
	B[0, i] = b.get_value()	
	time.sleep(2)
	
print 'Done with training the model!'

#save parameters for later use 
np.savez('./weights.npz', w=W)
np.savez('./bias.npz', b=B)
print 'Saved the parameters...'


"""
PREDICT THE ACCURACY OVER TRAINING SET
"""
H = 1.0 / (1.0 + T.exp(-T.dot(x, W) - B))
pred_vec = T.argmax(H, axis=1)
predict = function(inputs=[x], outputs=[H, pred_vec])


print 'Predicting accuracy over the training set: '
Hypo, predicted = predict(X_train)
print 'hypo shape: ', Hypo.shape
print 'predicted shape: ', predicted.shape
print '\n\nHypo: ', Hypo[0:20, :]
print '\n\npredicted: ', predicted[0:20]
print '\n\ntarget: ', Y_train[0:20]

pred_count = 0
for target, _pred in zip(Y_train, predicted):
	# print 'predicted: ', _pred, ' Target: ', target
	if target == _pred:
		pred_count += 1
print 'Accuracy over entire set of data is: ', (pred_count/5000.0) * 100