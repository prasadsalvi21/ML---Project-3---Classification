import tensorflow as tf
import os, cv2,pickle, time,gzip, numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
image_size = 28
num_labels = 10
num_channels = 1
learning_rate = 0.001
epochs = 1000
batch_size = 100
display_step = 100

def reformat_tf(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def SNN_TF():
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print("Loading MNIST Data.................")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    f.close()
    train_features, train_label = train_set
    validation_features, validation_label = valid_set
    test_features, test_label = test_set
    train_features = np.array(train_features)
    validation_features = np.array(validation_features)
    test_features = np.array(test_features)
    train_label = np.array(train_label)
    validation_label = np.array(validation_label)
    test_label = np.array(test_label)
    
    train_label=reformat(train_label) 
    validation_label=reformat(validation_label) 
    test_label=reformat(test_label) 

    
    print("Loading USPS Data.................")
    usps_data = []
    usps_label = []
    path_to_data = "./proj3_images/Numerals/"
    img_list = os.listdir(path_to_data)
    sz = (28,28)
    for i in range(10):
        label_data = path_to_data + str(i) + '/'
        img_list = os.listdir(label_data)
        for name in img_list:
            if '.png' in name:
                img = cv2.imread(label_data+name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized_img = resize_and_scale(img, sz, 255)
                usps_data.append(resized_img.flatten())
                usps_label.append(i)   
    usps_data = np.array(usps_data)
    usps_label= np.array(usps_label)   
    usps_dataset, usps_label = reformat_tf(usps_data, usps_label)             

    print("--------------------------------------")
    print("Training Started!")
    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
    
    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)+ (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        #total_batch = int(len(mnist.train.labels) / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(len(train_label) / batch_size)
            x_batches = np.array_split(train_features, total_batch)
            y_batches = np.array_split(train_label, total_batch)
            x_batches=np.array(x_batches)
            y_batches=np.array(y_batches)
            for i in range(total_batch):
                #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            if epoch % display_step == 0:    
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        
        print("Training complete!")
        print("--------------------------------------")
        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy MNIST Validation Data:", accuracy.eval({x: validation_features, y: validation_label}))
        print("Accuracy MNIST TEST Data:", accuracy.eval({x: test_features, y: test_label}))
        print("Accuracy USPS Data:", accuracy.eval({x: usps_data, y: usps_label}))
    	
        print("Loading USPS TEST Data.................")
        usps_data = []
        usps_label = []
        path_to_data = "./proj3_images/Test/"
        img_list = os.listdir(path_to_data)
        count=1
        j=9
        sz = (28,28)
        for name in sorted(img_list):
            if '.png' in name:
                if (count%150 !=0):
                        #print(name)
                        #print(j)
                        #name=name.split(_)[1]
                        img = cv2.imread(path_to_data+name)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized_img = resize_and_scale(img, sz, 255)
                        usps_data.append(resized_img.flatten())
                        usps_label.append(j)
                        count=count+1
                else:
                        #print(name)
                        #print(j)
                        count=1
                        j=j-1   
                        img = cv2.imread(path_to_data+name)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized_img = resize_and_scale(img, sz, 255)
                        usps_data.append(resized_img.flatten())
                        usps_label.append(j) 
        usps_data = np.array(usps_data)
        #print("USPS Data ",usps_data.shape)
        usps_label= np.array(usps_label)
        #print("USPS Labels ",reformat(usps_label).shape)
        
        usps_dataset, usps_label = reformat_tf(usps_data, usps_label)
        print("Accuracy USPS TEST Data:", accuracy.eval({x: usps_data, y: usps_label}))
        
