import os
import os.path as path

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

import matplotlib.pyplot as plt
from scipy.misc import toimage

import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print(y_test.shape)

    # normalize inputs from 0-255 to 0.0-1.0
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def floor(img):
    return np.uint8(img)


def to_canny(img):
    img = rgb2gray(img)
    edges = cv2.Canny(np.uint8(img), 32, 32)

    # plt.subplot(121), plt.imshow(img, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edges, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    return edges*255


def export_cifar10(category=0, type='train'):
    if not path.exists('dataset'):
        os.mkdir('dataset')

    if(type=='train'):
        color_train = None
        canny_train = None
        output_train = None
        for i in range(y_train.shape[0]):
            if (i % 100 == 0):
                print('{} / {}'.format(i, y_train.shape[0]))
            if (y_train[i][category] == 1):
                if (output_train is None):
                    color_train = np.array(
                        np.reshape(floor(img=x_train[i]), (1, 3, 32, 32)))
                    canny_train = np.array(
                        np.reshape(floor(img=to_canny(x_train[i])), (1, 1, 32, 32)))
                    output_train = np.array(np.reshape(y_train[i], (1, 10, )))
                else:
                    color_train = np.append(color_train,
                                             np.reshape(floor(img=x_train[i]), (1, 3, 32, 32)),
                                             axis=0)
                    canny_train = np.append(canny_train,
                                            np.reshape(floor(img=to_canny(x_train[i])), (1, 1, 32, 32)),
                                            axis=0)
                    output_train = np.append(output_train, np.reshape(y_train[i], (1, 10, )), axis=0)

        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        np.save('dataset/x_{}_color_{}.npy'.format(type, categories[category]), color_train)
        np.save('dataset/x_{}_canny_{}.npy'.format(type, categories[category]), canny_train)
        np.save('dataset/y_{}_generic_{}.npy'.format(type, categories[category]), output_train)

    else:
        color_test = None
        canny_test = None
        output_test = None
        for i in range(y_test.shape[0]):
            if (i % 100 == 0):
                print('{} / {}'.format(i, y_test.shape[0]))
            if (y_test[i][category] == 1):
                if (output_test is None):
                    color_test = np.array(
                        np.reshape(floor(img=x_test[i]), (1, 3, 32, 32)))
                    canny_test = np.array(
                        np.reshape(floor(img=to_canny(x_test[i])), (1, 1, 32, 32)))
                    output_test = np.array(np.reshape(y_test[i], (1, 10, )))
                else:
                    color_test = np.append(color_test,
                                             np.reshape(floor(img=x_test[i]), (1, 3, 32, 32)),
                                             axis=0)
                    canny_test = np.append(canny_test,
                                            np.reshape(floor(img=to_canny(x_test[i])), (1, 1, 32, 32)),
                                            axis=0)
                    output_test = np.append(output_test, np.reshape(y_test[i], (1, 10, )), axis=0)

        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        np.save('dataset/x_{}_color_{}.npy'.format(type, categories[category]), color_test)
        np.save('dataset/x_{}_canny_{}.npy'.format(type, categories[category]), canny_test)
        np.save('dataset/y_{}_generic_{}.npy'.format(type, categories[category]), output_test)


x_train, y_train, x_test, y_test = load_cifar10()
for i in range(10):
    export_cifar10(i, 'train')
for i in range(10):
    export_cifar10(i, 'test')