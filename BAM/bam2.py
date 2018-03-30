'''
Working example of Bidirectional associative memory
https://en.wikipedia.org/wiki/Bidirectional_associative_memory

NN-Team-2
'''

from keras.datasets import cifar10
from keras.utils import np_utils

import pprint

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
    print(X_train[0][0][0][1])


    data_pairs = [
        [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]],
        [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]],
        [[0, 0, 0, 1, 0, 1], [1, 1, 1, 0]]
    ]
    b = BAM(data_pairs)


    pp = pprint.PrettyPrinter(indent=4)
    print ('Matrix: ')
    pp.pprint(b.get_bam_matrix())
    print ('\n')
    print ('[1, 0, 1, 0, 1, 0] ---> ', b.get_assoc([1, 0, 1, 0, 1, 0]))
    print ('[1, 1, 1, 0, 0, 0] ---> ', b.get_assoc([1, 1, 1, 0, 0, 0]))
    print ('[0, 0, 0, 1, 1, 1] ---> ', b.get_assoc([0, 0, 0, 1, 1, 1]))
