import os, cv2, pickle, gzip, numpy as np
from PIL import Image
import SingleLayerNeuralNetwork_BP
import time
from datetime import timedelta

image_size = 28
num_labels = 10

'''
   Print Results
'''
def PrintResults(accuracy,displaystring):
      print(displaystring)
      print("Accuracy="+str(accuracy))

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

if __name__ == "__main__":
    start_time = time.monotonic()
    print("********************************************************************************")
    print("UBitName:prasadde")
    print("personNumber:50207353")
    print("UBitName:veerappa")
    print("personNumber:50247314")
    print("UBitName:sarahmul")
    print("personNumber:34508498")
    print("********************************************************************************")
    # Load the MNIST dataset
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
    #print("USPS Labels ",reformat(usps_label).shape)


    print('########################################################################')
    print('Single hidden layer neural network using Back Propogation')
    # call simple neural network subroutine
    wj, wk = SingleLayerNeuralNetwork_BP.simple_neural_network(train_features, train_label)
    # calculate argmax value for validation set
    accuracy,no_of_mismatches, err, no_mismatch = SingleLayerNeuralNetwork_BP.classification_rate_ann(train_features, train_label, wj, wk)
    print('Classification error rate for training  data is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for training set is ', no_mismatch)
    PrintResults(accuracy,"MNIST_Train_dataset_Sigle Layer Neural Network")
       
    accuracy,no_of_mismatches, err, no_mismatch = SingleLayerNeuralNetwork_BP.classification_rate_ann(validation_features, validation_label, wj, wk)
    print('Classification error rate for validation  data is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mi00smatches for validation set is ', no_mismatch)
    PrintResults(accuracy,"MNIST_validation_dataset_Sigle Layer Neural Network")
   
    accuracy,no_of_mismatches, err, no_mismatch = SingleLayerNeuralNetwork_BP.classification_rate_ann(test_features, test_label, wj, wk)
    print('Classification error rate for testing  data  is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for testing  set is ', no_mismatch)
    PrintResults(accuracy,"MNIST_test_dataset_Sigle Layer Neural Network")
       
    accuracy,no_of_mismatches, err, no_mismatch = SingleLayerNeuralNetwork_BP.classification_rate_ann(usps_data,usps_label, wj, wk)
    print('Classification error rate for USPS data set is is equal to ', no_of_mismatches, 'and cross entropy error is ', err)
    print('The number of mismatches for USPS data  set is ', no_mismatch)
    PrintResults(accuracy,"USPS_dataset_Sigle Layer Neural Network")
    end_time = time.monotonic()
    print("Time taken :","{:.3f}".format(end_time-start_time))
    
    