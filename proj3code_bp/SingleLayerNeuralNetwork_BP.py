import pickle
import random
import numpy
from os import listdir
from os.path import join
import os
from PIL import Image

no_of_labels = 10
no_of_features = 784
no_of_neurons = 1000
learning_rate = 0.001
reg_lambda = 0.01

def cross_entropy_error(probabilities, target_vector):
    error_sum = -numpy.sum(numpy.log(probabilities) * target_vector)
    return error_sum / no_of_labels


def classification_rate_ann(data, labels, _wj, _wk):
    mismatch = 0
    count=0
    for l in range(0, len(data)):
        x_i = data[l]
        _zj = numpy.dot(x_i, _wj)
        _a1 = sigmoid(_zj)
        _ak = numpy.dot(_a1, _wk)
        _ak = numpy.exp(_ak)

        sum_exp = numpy.sum(_ak)
        pr = _ak / sum_exp
        ind = numpy.argmax(pr)
        error_rate = cross_entropy_error(pr, pr)
        if ind != int(labels[l]):
            mismatch += 1
        if ind == int(labels[l]):
            count+=1
    accuracy = (float(count) / len(data)) * 100.0        
    return accuracy,mismatch/len(data), error_rate, mismatch

def sigmoid(a):
    b = 1 + numpy.exp(-a)
    return 1/b


def sigmoid_derivative(m):
    k = 1 - sigmoid(m)
    return sigmoid(m) * k

def simple_neural_network(_inputs, _labels):

    w_j = numpy.random.random((no_of_features, no_of_neurons))/numpy.sqrt(no_of_features)
    w_k = numpy.random.random((no_of_neurons, no_of_labels))/numpy.sqrt(no_of_neurons)
    bias_1 = numpy.ones((no_of_features, 1))
    bias_2 = numpy.ones((1, no_of_labels))
    w_j = numpy.c_[w_j, bias_1]
    w_k = numpy.vstack((w_k, bias_2))
    eta = 0.001  # learning rate

    for epochs in range(0, 25): 
        print("Epoch:", (epochs + 1))
        for index in range(0, len(_inputs)):  # the number of passes is 50,000 ,batch size is 1
            _input = _inputs[index]
            aj = numpy.dot(_input, w_j)
            zj = sigmoid(aj)
            ak = numpy.dot(zj, w_k)
            ak = numpy.exp(ak)
            _sum_exp = numpy.sum(ak)

            probabilities = ak / _sum_exp
            probabilities = numpy.array(probabilities).ravel()

            labels = numpy.zeros(shape=no_of_labels)
            labels[int(_labels[index])] = 1
            _error = cross_entropy_error(probabilities, labels)

            dk = probabilities - labels
            del_k = numpy.transpose(numpy.asmatrix(zj)) * dk
            x1 = numpy.dot(w_k, numpy.transpose(dk))
            temp = numpy.asmatrix(sigmoid_derivative(aj) * x1)  
            del_j = numpy.asmatrix(_input).T * temp
            w_j -= (eta * del_j)
            w_k -= (eta * del_k)    
    print(' The Single neural network cross entropy error for training set is ', _error * 100)
    return w_j, w_k

