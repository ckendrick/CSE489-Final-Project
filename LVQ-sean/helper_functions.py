import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


#Exctracts 8-bit RGB image from arr of size nxm starting at i0
def extract_img(arr, n,m, i0):
    im = np.zeros((n,m,3),dtype=np.uint8)
    for i in range(3): 
        for j in range(n):
            im[j,:,i] = arr[i0,i*n*m + j*m : i*n*m + (j+1)*m]
    return im

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