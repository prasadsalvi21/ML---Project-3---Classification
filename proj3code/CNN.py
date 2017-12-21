import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#Import USPS
import load_USPS 


#Generate random weights based on shape
def generateWeigths(shape):
	weight = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(weight)

#Generate bias weights based on shape
def generateBias(shape):
	bias = tf.constant(0.1, shape=shape)
	return tf.Variable(bias)

#Apply convolution to image x with filter W and move the strides 1 pixel
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Max pooling
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def CNN_TF():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

	# Convolutional Layer 1.
	filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
	num_filters1 = 32         # There are 16 of these filters.
	#num_filters1 = 16
	# Convolutional Layer 2.
	filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
	num_filters2 = 64         # There are 64 of these filters.
	#num_filters2 = 36
	# Fully-connected layer.
	fc_size = 1024             # Number of neurons in fully-connected layer.
	#fc_size = 128
	#Learning rate
	learning_rate = 1e-4
	epochs = 20000
	display = 1000
	
	# We know that MNIST images are 28 pixels in each dimension.
	img_size = 28
	# Images are stored in one-dimensional arrays of this length.
	img_size_flat = img_size * img_size
	# Tuple with height and width of images used to reshape arrays.
	img_shape = (img_size, img_size)
	# Number of colour channels for the images: 1 channel for gray-scale.
	num_channels = 1
	# Number of classes, one class for each of 10 digits.
	num_classes = 10
	
	#Placeholders to hold images and labels
	x = tf.placeholder(tf.float32, [None, img_size_flat])
	y_ =  tf.placeholder(tf.float32, [None, num_classes])
	
	#Convolution Layer 1, finds 32 features for each 5x5
	#Weights: patch_size, patch_size, i/p_channel, o/p_channel
	w_conv1 = generateWeigths([filter_size1, filter_size1, num_channels, num_filters1])
	b_conv1 = generateBias([num_filters1])
	
	#Reshape to shape any, width, height, colour_channel
	x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
	
	h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	
	#Convolution Layer 2, finds 64 features for each 5x5
	#Weights: patch_size, patch_size, i/p_channel, o/p_channel
	# 32 channels: 32*64 filters: Take 1,1 2,1.. 32,1 and apply to 1st pixel of each channel and sum to calculate on pixel. Repeat for depth 64
	w_conv2 =  generateWeigths([filter_size2, filter_size2, num_filters1, num_filters2])
	b_conv2 = generateBias([num_filters2])
	
	h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
	
	W_fc1 = generateWeigths([7 * 7 * num_filters2, fc_size])
	b_fc1 = generateBias([fc_size])
	
	#Reshaping 4D to 2D
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	#Drop out
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
	W_fc2 = generateWeigths([fc_size, num_classes])
	b_fc2 = generateBias([num_classes])
	
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)), tf.float32))
	
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for epoch in range(epochs):
	      batch = mnist.train.next_batch(50)
	      if epoch % display == 0:
	        cost = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
	        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost))
	      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
	    print('test accuracy %g' % accuracy.eval(feed_dict={
	        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
	
	    im, labels = load_USPS.load_USPS()
	    print('USPS accuracy %g' % accuracy.eval(feed_dict={
	        x: im, y_: labels, keep_prob: 1.0}))
	    im, labels = load_USPS.load_USPS_TEST()
	    print('USPS TEST accuracy %g' % accuracy.eval(feed_dict={
	        x: im, y_: labels, keep_prob: 1.0}))
	
	

	
	
	
	
	
	
