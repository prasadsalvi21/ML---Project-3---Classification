
# coding: utf-8

import numpy as np
import pickle, gzip, struct, cv2, sys,os,os.path, time
import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt
from sys import stderr
from PIL import Image

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels




def activation(X, W, b):
    return (X.dot(W) + b)

def softmax(z):
        e_x = np.exp(z - z.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out

def classlabels(z):
       return z.argmax(axis=1)


def cross_entropy(softmaxprob, y_target):
    return - np.sum(np.log(softmaxprob) * (y_target), axis=1)

###here, y_target is the one-hot representation


#change cost function if we want to add a regularization term.  Note cost function is in here if we want to do gradient descent or minibatch gradient descent; since it sums across all datapoints in batch
    
def cost(cross_entropy):
    return np.mean(cross_entropy) 



def one_hot(y, n_labels, dtype):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)


def minibatch_ix(rgen, n_batches, data):
            indices = np.arange(data.shape[0])
            indices = rgen.permutation(indices)
            if n_batches > 1:
                remainder = data.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for ix_batch in minis:
                yield ix_batch


def SGD_softmax(X, t, minibatches=1, random_seed=1, epochs=500, learning_rate=0.01):
    k_classes=np.max(t)+1 ##max of t=9 (+1 for 0)
    n_features=X.shape[1]
    #setting initial w, b parameters (w as random normal with mean 0, b as 0's)
    rgen = np.random.RandomState(random_seed)
    weights_shape=(n_features, k_classes)
    bias_shape=(k_classes,)
    dtype='float64'
    w = rgen.normal(loc=0.0, scale=0.01, size=weights_shape)
    b = np.zeros(shape=bias_shape)
    b.astype(dtype)
    w.astype(dtype)
    #create lists for cost and cross-entropy: equal for stochastic gradient descent
    cost_fit=[]
    cross_entropy_fit=[]
    y_enc = one_hot(y=t, n_labels=k_classes, dtype=np.float)
    for i in range(epochs):
            for ix in minibatch_ix(
                    rgen=rgen,
                    n_batches=minibatches,
                    data=t):
                 # activations, softmax and diff -> n_samples x n_classes:
                activation_input = activation(X[ix], w, b)
                softm = softmax(activation_input)
                diff = softm - y_enc[ix]

                # gradient -> n_features x n_classes
                grad = np.dot(X[ix].T, diff)

                # update in opp. direction of the cost gradient
                w -= (learning_rate * grad)
                b -= (learning_rate * np.sum(diff, axis=0))  ##derivative wrt b
            # compute cost of the whole epoch
            activation_input = activation(X, w, b)
            softm = softmax(activation_input)
            cross_ent = cross_entropy(softmaxprob=softm, y_target=y_enc)
            cost_out = cost(cross_ent)
            cost_fit.append(cost_out)
            cross_entropy_fit.append(cost_out)
    return cost_fit, softm, w, b, cross_entropy_fit



def predict_probability(X, w, b):
        activation_input = activation(X, w, b)
        softm = softmax(activation_input)
        return softm

def predict(X, w, b):
        prob = predict_probability(X, w, b)
        return classlabels(prob)


def accuracy(y_pred, t):
    Right=np.equal(y_pred, t)
    accuracy=np.sum(Right)/len(t)
    error=len(t)-np.sum(Right)
    misclasserror=error/len(t)
    return accuracy, error, misclasserror


def LogisticRegression():
        # Load the MNIST dataset
    print("Loading MNIST Data.................")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    f.close()
    
    #train_mnist=load_mnist(dataset="training", path="/Users/sarahmullin/Box Sync/3/Sarah")
    
    #load the dataset
    with gzip.open('mnist.pkl.gz','rb') as ff :
        u = pickle._Unpickler( ff )
        u.encoding = 'latin1'
        train, val, test = u.load()
    
    ###Relabelling MNIST data
    x_train=train[0]
    y_train=train[1]
    x_validate=val[0]
    y_validate=val[1]
    x_test=test[0]
    y_test=test[1]
    
    ###loading USPS data
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
    #print("USPS Data ",usps_data.shape)
    usps_label= np.array(usps_label)
    ##### Tune learning_rate using validation set
    # True learning_rate=0.01, 0.0001, 0.1
    
    #0.01
    #costfunction1, softmaxfunction1, w_train1, b_train1, cross1=SGD_softmax(X=x_train, t=y_train, minibatches=len(y_train), random_seed=1, epochs=1000, learning_rate=0.01)
    #Y_probability1=predict_probability(x_validate, w_train1, b_train1)
    #print(Y_probability1)
    #y_pred=classlabels(Y_probability1)
    #accuracy_01=accuracy(y_pred, y_validate)
    #print("Accuracy of validation set for eta=0.01", accuracy_01)
    #0.001
    #costfunction2, softmaxfunction2, w_train2, b_train2, cross2=SGD_softmax(X=x_train, t=y_train, minibatches=len(y_train), random_seed=1, epochs=1000, learning_rate=0.001)
    #Y_probability2=predict_probability(x_validate, w_train2, b_train2)
    #print(Y_probability2)
    #y_pred=classlabels(Y_probability2)
    #accuracy_001=accuracy(y_pred, y_validate)
    #print("Accuracy of validation set for eta=0.001", accuracy_001)
    #0.1
    #costfunction3, softmaxfunction3, w_train3, b_train3, cross3=SGD_softmax(X=x_train, t=y_train, minibatches=len(y_train), random_seed=1, epochs=1000, learning_rate=0.1)
    #Y_probability3=predict_probability(x_validate, w_train3, b_train3)
    #print(Y_probability3)
    #y_pred=classlabels(Y_probability3)
    #accuracy_1=accuracy(y_pred, y_validate)
    #print("Accuracy of validation set for eta=0.1", accuracy_1)
    
    
    
    ####best is learning_rate=0.001
    ####check accuracy
    
    #Y_probability=predict_proba(x_test, w_train, b_train)
    #print(Y_probability)
    #y_pred=to_classlabels(Y_probability)
    #accuracy(y_pred, y_test)
    
    
    costfunction_cross, softmaxfunction_cross, w_train_cross, b_train_cross, cross_ent_fit=SGD_softmax(X=x_train, t=y_train, minibatches=len(y_train), random_seed=1, epochs=1000, learning_rate=0.001)
    
    
    ###print weights and accuracy
    Y_probability=predict_probability(x_test, w_train_cross, b_train_cross)
    #print(Y_probability)
    y_pred=classlabels(Y_probability)
    accuracy_MNIST, mismatches_MNIST, miserror_MNIST=accuracy(y_pred, y_test)
    
    print("Accuracy of MNIST testing dataset based on training data", accuracy_MNIST)
    print("Number of Mismatches of MNIST testing dataset based on training data", mismatches_MNIST)
    print("Classification Error Rate of MNIST testing dataset based on training data", miserror_MNIST)
    
    ####plot cross entropy by iterations
    # evenly sampled time at 200ms intervals
    iterations = np.arange(start=1, stop=1001, step=1)
    #plt.plot(iterations, costfunction_cross)
    #plt.ylabel("Cost Function")
    #plt.xlabel("Epoch")
    #plt.show()
    
    #plt.plot(iterations, cross_ent_fit)
    #plt.ylabel("Cross-Entropy")
    #plt.xlabel("Epoch")
    #plt.show()
    
    
    ####Accuracy of the USPS data
    USPS_probability=predict_probability(usps_data, w_train_cross, b_train_cross)
    #print(USPS_probability)
    USPS_pred=classlabels(USPS_probability)
    accuracy_usps, mismatches_usps, miserror_usps=accuracy(USPS_pred, usps_label)
    print("Accuracy of USPS testing dataset based on MNIST training data", accuracy_usps)
    print("Number of Mismatches of USPS testing dataset based on MNIST training data", mismatches_usps)
    print("Classification Error Rate of USPS testing dataset based on training data", miserror_usps)

    

    
    
