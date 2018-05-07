# Simple CNN model for CIFAR-10
#
# NN-Team-2

import os
import os.path as path

import numpy
from keras.datasets import cifar10
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

# Simply orders the dataset for 'channels (3) first'
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
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


def tensorboard(log_dir):
    # starting tensorboard: tensorboard --logdir=run1:logs1/,run2:logs2/ --port 6006
    if not path.exists(log_dir):
        os.mkdir(log_dir)
    print('--- enabling TensorBoard')
    return TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

# tensorboard --logdirun0:logs_sgd10_0,run1:logs_sgd10_1,run2:logs_sgd10_2,run3:logs_sgd10_3,run4:logs_sgd10_4,run5:logs_sgd10_5,run6:logs_sgd10_6,run7:logs_sgd10_7 --port 6006
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
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=callbacks, verbose=False)

    return model

# tensorboard --logdi1e-4:logs_rms10_0,2e-4:logs_rms10_1,3e-4:logs_rms10_2,5e-5:logs_rms10_3,1e-5:logs_rms10_4 --port 6006
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

# tensorboard --logdi1e-4:logs_adam10_0,2e-4:logs_adam10_1,3e-4:logs_adam10_2,5e-5:logs_adam10_3,1e-5:logs_adam10_4 --port 6006
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

    # Compile model
    epochs = 75

    # List of parameters to try:
    tests = [(0.01, 0.90), (0.02, 0.90), (0.005, 0.90), (0.01, 0.90), (0.01, 0.85), (0.01, 0.80),
             (0.01, 0.75), (0.01, 0.95)]

    for i in range(0, tests.__len__()):
        print('--- testing: {} lrate and {} momentum'.format(tests[i][0], tests[i][1]))

        model = build_model(y_test.shape[1])

        model = train_sgd(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        filename = 'out/sdg10_{}.h5'.format(i)
        save(model, filename)

    ####################################################################################################################
    ## RMS

    # List of parameters to try:
    tests = [(0.0001), (0.0002), (0.0003), (0.00005), (0.00001)]

    epochs = 75
    for i in range(0, 3):
        print('--- testing: {} lrate'.format(tests[i]))

        model = build_model(y_test.shape[1])

        model = train_rms(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        evaluate(model, X_test, y_test)

        filename = 'out/rms10_{}.h5'.format(i)
        save(model, filename)

    epochs = 150
    for i in range(3, 4):
        print('--- testing: {} lrate'.format(tests[i]))

        model = build_model(y_test.shape[1])

        model = train_rms(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        evaluate(model, X_test, y_test)

        filename = 'out/rms10_{}.h5'.format(i)
        save(model, filename)

    epochs = 300
    for i in range(4, 5):
        print('--- testing: {} lrate'.format(tests[i]))

        model = build_model(y_test.shape[1])

        model = train_rms(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        evaluate(model, X_test, y_test)

        filename = 'out/rms10_{}.h5'.format(i)
        save(model, filename)

    ####################################################################################################################
    ## Adam

    # List of parameters to try:
    tests = [(0.0001), (0.0002), (0.0003), (0.00005)]

    epochs = 75
    for i in range(0, 3):
        print('--- testing: {} lrate'.format(tests[i]))

        model = build_model(y_test.shape[1])

        model = train_adam(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        evaluate(model, X_test, y_test)

        filename = 'out/adam10_{}.h5'.format(i)
        save(model, filename)

    epochs = 121
    for i in range(3, 4):
        print('--- testing: {} lrate'.format(tests[i]))

        model = build_model(y_test.shape[1])

        model = train_adam(model, X_train, y_train, X_test, y_test, None, tests[i], epochs)

        evaluate(model, X_test, y_test)

        filename = 'out/adam10_{}.h5'.format(i)
        save(model, filename)


if __name__ == '__main__':
    main()