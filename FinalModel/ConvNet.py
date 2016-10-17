import tensorflow as tf
import numpy as np
import time

rng = np.random

init_time = time.time()

#read data
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

#details of the network
learning_rate = 0.01
input_layer = 784
out_channels1 = 32
max_pooling1 = 14
out_channels2 = 64
max_pooling2 = 7
full_conn_layer = 1024
out_classes = 10

print '\nNetwork details...'
print 'Input size: ', input_layer
print 'Volume after 1st convolution: ', out_channels1
print 'Shape after max-pooling on 1st convolution: (%d x %d x %d)' \
      % (max_pooling1, max_pooling1, out_channels1)
print 'Volume after 2nd convolution: ', out_channels2
print 'Shape after max-pooling on 2nd convolution: (%d x %d x %d)' \
      % (max_pooling2, max_pooling2, out_channels2)
print 'Number of units in fully-connected layer: ', full_conn_layer
print 'Output layer units: ', out_classes

# graph input placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# model weights and biases
# 5 x 5 filters on 1 channel input producing 32 out channels
wc1_vals = tf.random_normal([5, 5, 1, out_channels1])
wc1 = tf.Variable(wc1_vals, name='wc1')

# 5 x 5 filters on 32 channel input producing 64 out channels
wc2_vals = tf.random_normal([5, 5, 32, out_channels2])
wc2 = tf.Variable(wc2_vals, name='wc2')

# fully connected layer weights
wf_vals = tf.random_normal([max_pooling2 * max_pooling2 * out_channels2, full_conn_layer])
wf1 = tf.Variable(wf_vals, name='wf1')

# fully connected layer weights
wo_vals = tf.random_normal([full_conn_layer, out_classes])
wo = tf.Variable(wo_vals, name='wo')

bc1 = tf.Variable(tf.random_normal([out_channels1]), name='bc1')
bc2 = tf.Variable(tf.random_normal([out_channels2]), name='bc2')
bf1 = tf.Variable(tf.random_normal([full_conn_layer]), name='bf1')
bo = tf.Variable(tf.random_normal([out_classes]), name='bo')

####################
# create the model #
####################
def model(x):
    stride1 = 1
    stride2 = 1

    pooling = 2
    print '\nModel details...'
    print 'stride for the 1st convolution: ', stride1
    print 'stride for the 2nd convolution: ', stride2

    # reshape the input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # apply convolution on input image with 5x5 filters with zero-padding
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, stride1, stride1, 1],
                         padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bc1)
    conv1 = tf.nn.relu(conv1)

    # apply max-pooling on the first convolution layer(32 channels) and reduce the size to 14x14
    max_pool1 = tf.nn.max_pool(conv1, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME')

    # 2nd convolution layer giving 64 output channels
    conv2 = tf.nn.conv2d(max_pool1, wc2, strides=[1, stride2, stride2, 1],
                         padding='SAME')
    conv2 = tf.nn.bias_add(conv2, bc2)
    conv2 = tf.nn.relu(conv2)

    # apply 2nd max-pooling
    max_pool2 = tf.nn.max_pool(conv2, ksize=[1, pooling, pooling, 1],
                               strides=[1, pooling, pooling, 1], padding='SAME')

    # fully connected layer
    # max_pool2 is of shape (7 x 7 x 64)
    # reshape max_pool2 to a 1-dimensional vector
    fc_input = tf.reshape(max_pool2, shape=[-1, max_pooling2 * max_pooling2 * out_channels2])

    hidden_activations = tf.nn.relu(tf.add(tf.matmul(fc_input, wf1), bf1))

    # output layer is a linear classifier
    output_activations = tf.add(tf.matmul(hidden_activations, wo), bo)

    return output_activations

pred = model(x)

#################
# cost function #
#################
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

# optimizer using stochastic gradient descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize all the variables
init_op = tf.initialize_all_variables()

#other details
batch_size = 128
training_epochs = 250
display_step = 20

####################
# launch the graph #
####################
print '\nLaunching the graph...'
with tf.Session() as sess:
    sess.run(init_op)

    total_batches = int(mnist.train.num_examples/batch_size)
    print 'Implementing batchwise stochastic gradient descent...'
    print 'batch size: ', batch_size
    print 'Total number of batches: ', total_batches

    for epoch in xrange(1, training_epochs):
        J = 0
        avg_cost = 0
        start = time.time()

        for batch in xrange(1, total_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
        
            sess.run(optimizer, feed_dict={x: x_batch, y: y_batch})
        
            J = sess.run(cost, feed_dict={x: x_batch, y: y_batch})
            avg_cost += J
        
            if batch % display_step == 0:
                print 'Epoch: ', epoch+1, ' Batch: ', batch, \
                      ' Batch avg cost: ', avg_cost/(batch_size * display_step)
        
        avg_cost /= total_batches
        print 'Epoch: ', epoch+1, '\tCost: ', avg_cost, '\tTime: ', time.time()-start
        # sess.run(optimizer, feed_dict={x:X_train, y:Y_train})
        # J = sess.run(cost, feed_dict={x:X_train, y:Y_train})

        # if epoch %  display_step == 0:
        #     print 'Iteration: ', epoch, '\tCost: ', J

    # done training!
    print 'Done training the model!'

    # validating the model
    J_train = sess.run(cost, feed_dict={x: X_train, y: Y_train})
    J_validate = sess.run(cost, feed_dict={x: X_validate, y: Y_validate})
    J_test = sess.run(cost, feed_dict={x: X_test, y: Y_test})

    print 'Final cost over training set: ', J_train
    print 'Final cost over validation set: ', J_validate
    print 'Final cost over test set: ', J_test

    # predict the hypothesis
    corr_pred = tf.equal(tf.argmax(y, dimension=1), tf.argmax(pred, dimension=1))
    accuracy = tf.reduce_mean(tf.cast(corr_pred, tf.float32))

    print '\nPredicting accuracy...'
    print 'Training accuracy: ', sess.run(accuracy, feed_dict={x: X_train, y: Y_train}) * 100
    print 'Validation accuracy: ', sess.run(accuracy, feed_dict={x: X_validate, y: Y_validate}) * 100
    print 'Test accuracy: ', sess.run(accuracy, feed_dict={x: X_test, y: Y_test}) * 100

print '\nTotal time taken: ', time.time() - init_time









