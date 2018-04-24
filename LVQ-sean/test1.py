from import_images import unpickle #unpacks cifar-10 images to dictionary
import numpy as np
import neurolab as nl #( https://pythonhosted.org/neurolab/ex_newlvq.html )
import pylab as pl
from time import clock


def create_target_array(labels_in, n_classes):
    target_len = len(labels_in)
    labels_out = np.zeros((target_len, n_classes), dtype=int)
    
    for i in range(target_len):
        class_i = labels_in[i]
        labels_out[i][class_i] = 1
    return labels_out

#get normalized percentage-based distribution of each of the classes associated with
#labels_in from 0 to n_classes
def get_norm_dist(labels_in, n_classes):
    h = np.histogram(labels_in, range(0,n_classes+1)) #data histogram
    dist = list(h[0])
#    return list(dist / np.linalg.norm(dist))
    nv = 1.0 * sum(dist)
    return dist / nv  

start_time =  clock()
ts1 = unpickle('./images/data_batch_1')

data1 = ts1["data"]
print(data1.shape) #print size of nxm array
labels1 = ts1["labels"]
print(len(labels1)) #print size of n-length list
n_classes1 = max(labels1) + 1
labels_nl = create_target_array(labels1, n_classes1)
distro1 = get_norm_dist(labels1, n_classes1) #The percentage distribution of
#items per class

# Create train samples
""" input = np.array([[-3, 0], [-2, 1], [-2, -1], [0, 2], [0, 1], [0, -1], [0, -2], 
                                                        [2, 1], [2, -1], [3, 0]])
target = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], 
                                                        [1, 0], [1, 0], [1, 0]])
"""

# Create LVQ network from image data; ? neurons in input layer, 10 neourons in output layer
# to map each of the 10k images to one of the 10 output categories

n_neurons_in = 100
net = nl.net.newlvq(nl.tool.minmax(data1), n_neurons_in, distro1)

stop_time = clock()
print("Preparation took %0.8f seconds" %(stop_time-start_time)) 

#Train network
start_time =  clock()
error = net.train(data1, labels_nl, epochs=100, goal=0.01)
stop_time = clock()
print("Training took %0.8f seconds" %(stop_time-start_time)) 

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
    


"""
net = nl.net.newlvq(nl.tool.minmax(data1), 4, [.6, .4])
# Train network
error = net.train(input, target, epochs=1000, goal=-1)

# Plot result
xx, yy = np.meshgrid(np.arange(-3, 3.4, 0.2), np.arange(-3, 3.4, 0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
i = np.concatenate((xx, yy), axis=1)
o = net.sim(i)
grid1 = i[o[:, 0]>0]
grid2 = i[o[:, 1]>0]

class1 = input[target[:, 0]>0]
class2 = input[target[:, 1]>0]

pl.plot(class1[:,0], class1[:,1], 'bo', class2[:,0], class2[:,1], 'go')
pl.plot(grid1[:,0], grid1[:,1], 'b.', grid2[:,0], grid2[:,1], 'gx')
pl.axis([-3.2, 3.2, -3, 3])
pl.xlabel('Input[:, 0]')
pl.ylabel('Input[:, 1]')
pl.legend(['class 1', 'class 2', 'detected class 1', 'detected class 2'])
pl.show()
"""