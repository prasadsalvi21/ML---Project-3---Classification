import os, cv2, pickle, gzip, numpy as np
from PIL import Image
import SingleLayerNeuralNetwork_TF
import LogisticRegression_py
import CNN
import time
from datetime import timedelta

if __name__ == "__main__":
    print("********************************************************************************")
    print("UBitName:prasadde")
    print("personNumber:50207353")
    print("UBitName:veerappa")
    print("personNumber:50247314")
    print("UBitName:sarahmul")
    print("personNumber:34508498")
    print("********************************************************************************")
    start_time = time.monotonic()
    print('Multi-class Logistic Regression')
    LogisticRegression_py.LogisticRegression()
    end_time = time.monotonic()
    print("Time taken for Multi-class Logistic Regression calculation :","{:.3f}".format(end_time-start_time))
    print("********************************************************************************")
    start_time = time.monotonic()
    print('Single hidden layer neural network using Tensor Flow')
    SingleLayerNeuralNetwork_TF.SNN_TF()
    end_time = time.monotonic()
    print("Time taken for SNN calculation :","{:.3f}".format(end_time-start_time))
    print("********************************************************************************")
    start_time = time.monotonic()
    print('Convolutional neural network using Tensor Flow')
    CNN.CNN_TF()
    end_time = time.monotonic()
    print("Time taken for CNN calculation :","{:.3f}".format(end_time-start_time))
    print("********************************************************************************")
