import tensorflow as tf
import numpy as np
import time

rng = np.random

init_time = time.time()
# read data
print 'Reading data...'
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/kv/MNIST_data/", one_hot=True)

# (55000, 784)
X_train = mnist.train.images
Y_train = mnist.train.labels

# (5000, 784)
X_validate = mnist.validation.images
Y_validate = mnist.validation.labels

# (10000, 784)
X_test = mnist.test.images
Y_test = mnist.test.labels

# details about the network
input_layer = 784  # 28 * 28 images flattened
hidden_layer_1 = 256
hidden_layer_2 = 256
output_layer = 10

print '\nNetwork details...'
print 'Input size: ', input_layer
print 'Hidden layer 1 units: ', hidden_layer_1
print 'Hidden layer 2 units: ', hidden_layer_2
print 'Output layer units: ', output_layer

# graph input
x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

# model weights
print '\nInitialising random weights and biases...'
w_hidden1_vals = tf.random_normal([input_layer, hidden_layer_1])
b_hidden1_vals = tf.random_normal([hidden_layer_1])
w_hidden1 = tf.Variable(w_hidden1_vals, name='hidden1_weights')
b_hidden1 = tf.Variable(b_hidden1_vals, name='hidden1_bias')

w_hidden2_vals = tf.random_normal([hidden_layer_1, hidden_layer_2])
b_hidden2_vals = tf.random_normal([hidden_layer_2])
w_hidden2 = tf.Variable(w_hidden2_vals, name='hidden2_weights')
b_hidden2 = tf.Variable(b_hidden2_vals, name='hidden2_bias')


w_output_vals = tf.random_normal([hidden_layer_2, output_layer])
b_output_vals = tf.random_normal([output_layer])
w_output = tf.Variable(w_output_vals, name='output_weights')
b_output = tf.Variable(b_output_vals, name='output_bias')

# model for a multi-layer-perceptron with two hidden layers
# ReLU activations for the first hidden layer
hidden1_activations = tf.nn.relu(tf.add(tf.matmul(x, w_hidden1), b_hidden1))

#ReLU activations for the second hidden layer
hidden2_activations = tf.nn.relu(tf.add(tf.matmul(hidden1_activations, w_hidden2), b_hidden2))

# linear activations for the output layer
output_activations = tf.add(tf.matmul(hidden2_activations, w_output), b_output)

# using negative log-likelihood as the cost function
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output_activations), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_activations, y))

# using gradient descent optimizer
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize variables
init_op = tf.initialize_all_variables()

# other variables
batch_size = 100
cost_vec = []
training_epochs = 15

# launch the graph
print '\nLaunching the graph...'
with tf.Session() as sess:
	sess.run(init_op)

	total_batches = int(mnist.train.num_examples / batch_size)
	print 'Implementing batchwise stochastic gradient descent...'
	print 'batch size: ', batch_size
	print 'Total number of batches: ', total_batches

	for epoch in xrange(training_epochs):
		avg_cost = 0
		J = 0
		start = time.time()

		for batch in xrange(total_batches):
			x_batch, y_batch = mnist.train.next_batch(batch_size)

			# run a single step of gradient descent
			sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})

			# compute the cost over each batch
			J = sess.run(cost, feed_dict={x: x_batch, y: y_batch})

		avg_cost += J

		avg_cost /= total_batches
		print 'Epoch: ', epoch + 1, ' Cost: ', avg_cost, ' Time: ', time.time() - start
		cost_vec.append(avg_cost)

	print '\nDone training the model...'

	# validating the model
	J_train = sess.run(cost, feed_dict={x: X_train, y: Y_train})
	J_validate = sess.run(cost, feed_dict={x: X_validate, y: Y_validate})
	J_test = sess.run(cost, feed_dict={x: X_test, y: Y_test})

	print 'Final cost over training set: ', J_train
	print 'Final cost over validation set: ', J_validate
	print 'Final cost over test set: ', J_test

	# predict the hypothesis
	corr_pred = tf.equal(tf.argmax(y, dimension=1), tf.argmax(output_activations, dimension=1))
	accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float64))

	print '\nPredicting accuracy...'
	print 'Training accuracy: ', sess.run(accuracy, feed_dict={x: X_train, y: Y_train}) * 100
	print 'Validation accuracy: ', sess.run(accuracy, feed_dict={x: X_validate, y: Y_validate}) * 100
	print 'Test accuracy: ', sess.run(accuracy, feed_dict={x: X_test, y: Y_test}) * 100

	print '\nSaving the parameters...'
	np.savez('./Params/w_hidden1', w_hidden1=sess.run(w_hidden1))
	np.savez('./Params/b_hidden1', b_hidden1=sess.run(b_hidden1))
	np.savez('./Params/w_hidden2', w_hidden2=sess.run(w_hidden2))
	np.savez('./Params/b_hidden2', b_hidden2=sess.run(b_hidden2))
	np.savez('./Params/w_output', w_output=sess.run(w_output))
	np.savez('./Params/b_output', b_output=sess.run(b_output))

print '\nTotal time taken: ', time.time() - init_time
