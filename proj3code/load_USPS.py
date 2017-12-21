import tensorflow as tf
import cv2
import os,pickle, time,gzip, numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
image_size = 28
num_labels = 10
num_channels = 1

def reformat_tf(dataset, labels):
    #dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def load_USPS():
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
    
    usps_dataset, usps_label = reformat_tf(usps_data, usps_label)
    return usps_dataset, usps_label

def load_USPS_TEST():
    print("Loading USPS Data.................")
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
    usps_label= np.array(usps_label)
    usps_dataset, usps_label = reformat_tf(usps_data, usps_label)
    return usps_dataset, usps_label  
