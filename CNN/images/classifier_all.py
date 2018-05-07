# Simple CNN model for CIFAR-10
#
# NN-Team-2

import os
import os.path as path

from scipy import ndimage
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard

# Compile model
epochs = 75

# List of parameters to try:
tests = [(0.01, 0.90), (0.02, 0.90), (0.005, 0.90), (0.01, 0.90), (0.01, 0.85), (0.01, 0.80),
         (0.01, 0.75), (0.01, 0.95)]

# Simply orders the dataset for 'channels (3) first'
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def load_data():

    (X_train, y_train), (X_test, y_test) = import_images('../../Datasets/Images/export')

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test

def import_images(root_path):
    X = []
    y = []
    k = 0

    for root, dirs, files in os.walk(root_path, topdown=True):

        if dirs is not None:
            for dir in dirs:

                for root, dirs, files in os.walk('{}/{}'.format(root_path, dir)):
                    for i, f in enumerate(files):
                        img = ndimage.imread('{}/{}/{}'.format(root_path, dir, f))

                        if img.shape == (200, 200, 3):
                            X.append(img.reshape(3, 200, 200))
                            y.append(k)
                            # print('{}: {}/{}'.format(k, i, files.__len__()))

                k += 1

    print(np.array(X).shape)
    print(np.array(y).shape)

    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))





def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 200, 200), activation='relu', padding='same'))
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

# tensorboard --logdir=sgd_best:logs_sgdIM,rms_best:logs_rmsIM,adam_best:logs_adamIM --port 6006
def tensorboard(log_dir):
    # starting tensorboard: tensorboard --logdir=run1:logs1/,run2:logs2/ --port 6006
    if not path.exists(log_dir):
        os.mkdir(log_dir)
    print('--- enabling TensorBoard')
    return TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)


def train_sgd(model, X_train, y_train, X_test, y_test, log_dir, tests, epochs):

    lrate = tests[0]
    momentum = tests[1]
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=momentum, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit the model
    if log_dir is None:
        callbacks = []
    else:
        callbacks = [tensorboard(log_dir)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=True)

    return model

def train_rms(model, X_train, y_train, X_test, y_test, log_dir, tests, epochs):

    rms = RMSprop(lr=tests)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

    # Fit the model
    if log_dir is None:
        callbacks = []
    else:
        callbacks = [tensorboard(log_dir)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=False)

    return model

def train_adam(model, X_train, y_train, X_test, y_test, log_dir, tests, epochs):

    adam = Adam(lr=tests)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # Fit the model
    if log_dir is None:
        callbacks = []
    else:
        callbacks = [tensorboard(log_dir)]
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=False)

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

    X_train, y_train, X_test, y_test = load_data()

    ####################################################################################################################
    ## SGD

    # epochs = 75
    #
    # model = build_model(y_test.shape[1])
    #
    # model = train_sgd(model, X_train, y_train, X_test, y_test, 'logs_sgdIM', (0.01, 0.95), epochs)
    #
    # evaluate(model, X_test, y_test)
    #
    # filename = 'out/sgdIM.h5'
    # save(model, filename)

    ####################################################################################################################
    ## RMS

    epochs = 400

    model = build_model(y_test.shape[1])

    model = train_rms(model, X_train, y_train, X_test, y_test, 'logs_rmsIM', 0.00001, epochs)

    evaluate(model, X_test, y_test)

    filename = 'out/rmsIM.h5'
    save(model, filename)

    ####################################################################################################################
    ## Adam

    epochs = 150

    model = build_model(y_test.shape[1])

    model = train_adam(model, X_train, y_train, X_test, y_test, 'logs_adamIM', 0.0003, epochs)

    evaluate(model, X_test, y_test)

    filename = 'out/adamIM.h5'
    save(model, filename)


if __name__ == '__main__':
    main()