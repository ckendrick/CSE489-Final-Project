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


def export_cifar10(category=0):
    if not path.exists('dataset'):
        os.mkdir('dataset')

    filter_train = None
    canny_train = None
    for i in range(y_train.shape[0]):
        if (i % 100 == 0):
            print('{} / {}'.format(i, y_train.shape[0]))
        if (y_train[i][category] == 1):
            if (filter_train is None and canny_train is None):
                filter_train = np.array(
                    np.reshape(floor(img=x_train[i]), (1, 32, 32, 3)))
                canny_train = np.array(
                    np.reshape(floor(img=to_canny(x_train[i])), (1, 32, 32, 1)))
            else:
                filter_train = np.append(filter_train,
                                         np.reshape(floor(img=x_train[i]), (1, 32, 32, 3)),
                                         axis=0)
                canny_train = np.append(canny_train,
                                        np.reshape(floor(img=to_canny(x_train[i])), (1, 32, 32, 1)),
                                        axis=0)

    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    np.save('dataset/x_train_color_{}.npy'.format(categories[category]), filter_train)
    np.save('dataset/x_train_canny_{}.npy'.format(categories[category]), canny_train)


x_train, y_train, x_test, y_test = load_cifar10()
for i in range(10):
    export_cifar10(i)