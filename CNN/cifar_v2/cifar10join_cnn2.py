# Simple CNN model for CIFAR-100
#
# NN-Team-2

import os
import os.path as path

import numpy
from keras.models import Sequential, Model
from keras.layers import *
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import *
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
from scipy.misc import toimage

# Compile model
epochs = 1
lrate = 0.01

# Simply orders the dataset for 'channels (3) first'
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_data(type):

    X_train = None
    y_train = None
    for i in range(len(categories)):
        x_train_color = np.load('../../Datasets/CIFAR10_canny/dataset/x_train_{}_{}.npy'.format(type, categories[i]))
        y_train_color = np.load('../../Datasets/CIFAR10_canny/dataset/y_train_generic_{}.npy'.format(categories[i]))

        if (X_train is None):
            X_train = x_train_color
            y_train = y_train_color
        else:
            X_train = np.append(X_train, x_train_color, axis=0)
            y_train = np.append(y_train, y_train_color, axis=0)

    X_test = None
    y_test = None
    for i in range(len(categories)):
        X_test_color = np.load('../../Datasets/CIFAR10_canny/dataset/x_test_{}_{}.npy'.format(type, categories[i]))
        y_test_color = np.load('../../Datasets/CIFAR10_canny/dataset/y_test_generic_{}.npy'.format(categories[i]))

        if (X_test is None):
            X_test = X_test_color
            y_test = y_test_color
        else:
            X_test = np.append(X_test, X_test_color, axis=0)
            y_test = np.append(y_test, y_test_color, axis=0)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return X_train, y_train, X_test, y_test


def complex_cnn(num_classes, channels):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(channels, 32, 32), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def tensorboard():
    # starting tensorboard: tensorboard --logdir=run1:logs/ --port 6006
    if not path.exists('logs'):
        os.mkdir('logs')
    print('--- enabling TensorBoard')
    return TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)


def train(model, X_train, y_train, X_test, y_test):
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=[tensorboard()])

    return model


def evaluate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def load(model, filename):
    if not path.exists('out'):
        os.mkdir('out')

    if path.exists(filename):
        model.load_weights(filename)

    return model


def save(model, filename):
    if not path.exists('out'):
        os.mkdir('out')

    model.save_weights(filename)


def main():
    filename = 'out/cifar10split_complex_cnn.h5'

    X_train_color, y_train_color, X_test_color, y_test_color = load_data('color')
    X_train_canny, y_train_canny, X_test_canny, y_test_canny = load_data('canny')

    # Build two models, one for color image and one for canny-edge
    model_final = complex_cnn(y_test_color.shape[1], 3)
    # model_canny = complex_cnn(y_test_canny.shape[1], 1)
    # model_merge = Add()([model_color.output, model_canny.output])
    # model_merge = Dense(y_test_color.shape[1], activation='softmax')(model_merge)
    # model_final = Model([model_color.input, model_canny.input], model_merge)

    model = load(model_final, filename)
    model.summary()

    model = train(model, X_train_color, y_train_color, X_test_color, y_test_color)

    evaluate(model, X_test_color, y_test_color)

    save(model, filename)


if __name__ == '__main__':
    main()