
# coding: utf-8

# In[1]:

import numpy as np
import pickle, gzip, struct, cv2, sys,os,os.path, time
import tensorflow as tf
import random as ran
import matplotlib.pyplot as plt
from sys import stderr
from PIL import Image


# In[2]:

##############For Bayesian Multinomial Logistic Regression with a Normal Prior###############
###Reference: Madigan et al. Bayesian Multinomial Logistic Regression for Author Identification (2005)

######our data distribution is inherently multinomial
#with conditional probability modelled as P(y_k=1|X,W)=softmax(X.dot(W) + b)
#therefore P(y|W)=multinomial(P(y_k=1|X,W)

#we will take a gaussian prior distributions s.t. P(w)=N(0, sigma^2_kj) 

#Therefore, our posterior distribution P(W|x,y, sigma^2)=P(y|W)*P(w)
#For simplicity, i will define all sigma^2_kj associated with each w_kj as constant .5

#w_MAP is then solved by optimizing the negative log-likelihood of the posterior distributions;
#w_MAP_hat=argmax (w.r.t.w) P(w|D)=arg max {log P(w)+log P(D|w)}
# here, w_map is equivalent to w_MLE with regularizer

#Note: the posterior distribution of a multinomial and prior normal distribution with 0 mean and constant sigma is the same as doing multinomial logistic regression with ridge regularization (L2)


# In[3]:


def resize_and_scale(img, size, scale):
    img = cv2.resize(img, size)
    return 1 - np.array(img, "float32")/scale

def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels


# In[4]:

def activation(X, W, b):
    return (X.dot(W) + b)


# In[5]:

def softmax(z):
        e_x = np.exp(z - z.max(axis=1, keepdims=True))
        pr = e_x / e_x.sum(axis=1, keepdims=True)
        return pr
    
    


# In[6]:

def classlabels(z):
       return z.argmax(axis=1)


# In[7]:

def negloglikemultinomial(softmaxprob, y_target):
    return - np.sum(np.log(softmaxprob) * (y_target), axis=1)

###here, y_target is the one-hot representation; the first part of this is the multinomial distribution and the second part is the prior


# In[8]:

def negloglikeprior(sigma,w):
    return (1/(2*sigma) * np.sum(w ** 2))


# In[9]:

#change cost function if we want to add a regularization term.  Note cost function is in here if we want to do gradient descent or minibatch gradient descent; since it sums across all datapoints in batch
    
def negloglikposterior_distribution(negloglikemultinomial, negloglikprior):
    posterior=negloglikemultinomial + negloglikprior
    return np.mean(posterior) 


# In[10]:

def one_hot(y, n_labels, dtype):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)


# In[11]:

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


# In[12]:

def SGD_bayesiansoftmax(X, t, sigma, minibatches=1, random_seed=1, epochs=500, learning_rate=0.01):
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
    #create list for posterior distribution sampling using gradient descent optimization
    posterior_fit=[]
    y_enc = one_hot(y=t, n_labels=k_classes, dtype=np.float)
    for i in range(epochs):
            for ix in minibatch_ix(
                    rgen=rgen,
                    n_batches=minibatches,
                    data=t):
                 # gradient of logliklihood of P(Y|w,x) 
                activation_input = activation(X[ix], w, b)
                pr = softmax(activation_input)
                diff = pr - y_enc[ix]
                grad = np.dot(X[ix].T, diff)
                #gradient of prior distribution
                grad_prior=sigma*w
                # maximimize posterior distribution with respect to w to find w_MAP 
                w -= ((learning_rate * grad) +(learning_rate *grad_prior)) 
                b -= (learning_rate * np.sum(diff, axis=0))  ##derivative wrt b
            # compute cost of the whole epoch
            activation_input = activation(X, w, b)
            pr = softmax(activation_input)
            data_fit = negloglikemultinomial(softmaxprob=pr, y_target=y_enc)
            prior_fit=negloglikeprior(sigma,w)
            post_out =negloglikposterior_distribution(data_fit, prior_fit)
            posterior_fit.append(post_out)
    return posterior_fit, pr, w, b


# In[13]:

def predict_probability(X, w, b):
        activation_input = activation(X, w, b)
        softm = softmax(activation_input)
        return softm

def predict(X, w, b):
        prob = predict_probability(X, w, b)
        return classlabels(prob)


# In[14]:

def accuracy(y_pred, t):
    Right=np.equal(y_pred, t)
    accuracy=np.sum(Right)/len(t)
    error=len(t)-np.sum(Right)
    misclasserror=error/len(t)
    return accuracy, error, misclasserror


# In[ ]:

def BayesLogisticRegression():
    # Load the MNIST dataset
    print("************ Loading MNIST Data ************")
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
    print("************ Loading USPS Data ************")
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


    posterior_fit, pr_bayes, w_bayes, b_bayes=SGD_bayesiansoftmax(X=x_train, t=y_train, sigma=2, minibatches=1, random_seed=1, epochs=100, learning_rate=0.001)


    ####plot cross entropy by iterations
    # evenly sampled time at 200ms intervals
    iterations = np.arange(start=1, stop=101, step=1)
    #plt.plot(iterations, costfunction_cross)
    #plt.ylabel("Cost Function")
    #plt.xlabel("Epoch")
    #plt.show()

    plt.plot(iterations, posterior_fit)
    plt.ylabel("NegLogLikelihood of Posterior Distribution")
    plt.xlabel("Epoch")
    plt.show()
    
    print("W_map:",w_bayes, "Bias_map", b_bayes)


# In[ ]:

BayesLogisticRegression()

