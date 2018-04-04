'''
Working example of Bidirectional associative memory
Using MNIST - only 3 images are able to be stored...

NN-Team-2
'''

from keras.datasets import cifar10
from keras.utils import np_utils

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import toimage


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def plot_encoding(X_train):
    plt.figure(1)
    ax = plt.subplot(121)
    ax.title.set_text('grayscale')
    gray = rgb2gray(X_train[0])
    print(gray)
    plt.imshow(toimage(np.asarray(gray.astype(int))), cmap='gray')

    ax = plt.subplot(122)
    ax.title.set_text('basic encoding')
    for x in range(gray.shape[0]):
        for y in range(gray.shape[1]):
            if gray[x, y] > 255 / 2.5:
                gray[x, y] = 255
            else:
                gray[x, y] = 0
    plt.imshow(toimage(np.asarray(gray)), cmap='gray')

    plt.show()


def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


class BAM(object):
    def __init__(self, data):
        self.AB = []
        # store associations in bipolar form to the array
        for item in data:
            self.AB.append(
                [self.__l_make_bipolar(item[0]),
                 self.__l_make_bipolar(item[1])]
            )
        self.len_x = len(self.AB[0][1])
        self.len_y = len(self.AB[0][0])
        # create empty BAM matrix
        self.M = [[0 for x in range(self.len_x)] for x in range(self.len_y)]
        # compute BAM matrix from associations
        self.__create_bam()

    def __create_bam(self):
        '''Bidirectional associative memory'''
        for assoc_pair in self.AB:
            X = assoc_pair[0]
            Y = assoc_pair[1]
            # calculate M
            for idx, xi in enumerate(X):
                for idy, yi in enumerate(Y):
                    self.M[idx][idy] += xi * yi

    def get_assoc(self, A):
        '''Return association for input vector A'''
        A = self.__mult_mat_vec(A)
        return self.__threshold(A)

    def get_bam_matrix(self):
        '''Return BAM matrix'''
        return self.M

    def __mult_mat_vec(self, vec):
        '''Multiply input vector with BAM matrix'''
        v_res = [0] * self.len_x
        for x in range(self.len_x):
            for y in range(self.len_y):
                v_res[x] += vec[y] * self.M[y][x]
        return v_res

    def __threshold(self, vec):
        '''Transform vector to [0, 1]'''
        ret_vec = []
        for i in vec:
            if i < 0:
                ret_vec.append(0)
            else:
                ret_vec.append(1)
        return ret_vec

    def __l_make_bipolar(self, vec):
        '''Transform vector to bipolar form [-1, 1]'''
        ret_vec = []
        for item in vec:
            if item == 0:
                ret_vec.append(-1)
            else:
                ret_vec.append(1)
        return ret_vec


if __name__ == "__main__":

    X_train, y_train, _, _ = load_data()

    data_pairs = []

    j = 5
    for i in range(j):
        gray = rgb2gray(X_train[i, :])
        gray = gray.reshape(gray.shape[0] * gray.shape[0])

        for k, item in enumerate(gray):
            if gray[k] > 255/2.5:
                gray[k] = 1
            else:
                gray[k] = 0
        data_pairs.append([np.asarray(gray.astype(int)), y_train[i]])

    plot_encoding(X_train)

    b = BAM(data_pairs)

    print('\n')
    for i in range(j):
        print('X_train[{},:] ---> '.format(i), b.get_assoc(data_pairs[i][0]))
        print('       expecting: ', y_train[i])
        print('')

    print('')
    print('Untrained:')
    print('')

    gray = rgb2gray(X_train[j, :])
    gray = gray.reshape(gray.shape[0] * gray.shape[0])

    for k, item in enumerate(gray):
        if gray[k] > 255 / 2:
            gray[k] = 1
        else:
            gray[k] = 0
    print('X_train[{},:] ---> '.format(j), b.get_assoc(np.asarray(gray.astype(int))))
    print('       expecting: ', y_train[j])
