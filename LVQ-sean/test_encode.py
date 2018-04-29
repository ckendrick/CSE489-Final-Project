import numpy as np
import helper_functions as hf #unpacks cifar-10 images to dictionary
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import clock
import neurolab as nl #( https://pythonhosted.org/neurolab/ex_newlvq.html )
import pylab as pl

## constants
nFeatures = 128 #argument for nfeatures in feature detector
num_features = 100 #number of features per image
feature_size = 32 #feature size, bytes
feature_vec_size = num_features*feature_size

img_resize = 512 #dimension to resize image to (px)

n_neurons_in = 15 #number of neurons in first layer of algorithm
max_epochs = 50
error_goal = 0.1 #target training error


## Run loop to extract and encode each image individually

ts1 = hf.unpickle('./images/data_batch_1')
imgs1 = ts1["data"] #images 1
labels1 = ts1["labels"]
dataset_size = imgs1.shape[0]

data1 = np.zeros((dataset_size,feature_vec_size),dtype=np.uint8) #create empty encoding array

orb = cv.ORB_create(nfeatures=128) #create empty ORB feature recognition object

#for each image, do feature extraction and send features to data1 array
start_time =  clock()
for i in range(dataset_size):
    tmp_img = hf.extract_img(imgs1,32,32,i) #Exctract an image
    tmp_img = cv.cvtColor(tmp_img,cv.COLOR_RGB2GRAY) #change image to grayscale
    tmp_img = cv.resize(tmp_img,(img_resize,img_resize),interpolation=0) #resize image
    
    #Find keypoints and descriptors
    kp = orb.detect(tmp_img,None)
    kp,des = orb.compute(tmp_img, kp)
    if type(des)==type(None):
        print("Warning: at i=%d no features found" %i)
        kp = []
        des = []
    n_des = min(len(des),num_features) #number of descriptors to use - truncate if too many
    
    for j in range(n_des):
        start_i = j*feature_size
        end_i = (j+1)*feature_size
        data1[i,start_i:end_i] = des[j] #stack each descriptor linearly
stop_time = clock()
print("Feature Recognition+Encoding took %0.2f seconds" %(stop_time-start_time)) 


## Finish building components required to run LVQ algorithm
n_classes1 = max(labels1) + 1 #number of classes
labels_nl = hf.create_target_array(labels1, n_classes1) #reform labels to fit algorithm-required format
distro1 = hf.get_norm_dist(labels1, n_classes1) #The percentage distribution of items per class

#Build and train network
net = nl.net.newlvq(nl.tool.minmax(data1), n_neurons_in, distro1)

start_time =  clock()
error = net.train(data1, labels_nl, epochs=max_epochs, goal=error_goal)
stop_time = clock()
print("Training took %0.2f seconds" %(stop_time-start_time)) 

#Test network accuracy on some dataset
num_correct = 0
num_tested = 0
num_total = data1.shape[0]
num_chunks = 50
for i in range(num_chunks):
    min_j = i*num_total/num_chunks
    next_min_j = (i+1)*num_total/num_chunks
    chunk_size = next_min_j - min_j
    test_chunk = data1[min_j:next_min_j]
    test_results = net.sim(test_chunk)
    target_results = labels_nl[min_j:next_min_j]
    correctness_list = [(target_results[i] == test_results[i]).all() for i in range(chunk_size)]
    num_correct += sum(correctness_list)
    num_tested += chunk_size
    print ("%d/%d items categorized correctly\t %0.3f %% correct" %(num_correct, num_tested, 100.0*num_correct/num_tested))