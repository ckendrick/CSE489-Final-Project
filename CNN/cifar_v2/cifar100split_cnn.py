# Simple CNN model for CIFAR-100
#
# NN-Team-2

import os
import os.path as path

import numpy
from keras.datasets import cifar100
from keras.models import Sequential
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
epochs = 50
lrate = 0.01

# Simply orders the dataset for 'channels (3) first'
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def load_data(type):

    if type == 'color':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        # normalize inputs from 0-255 to 0.0-1.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        return X_train, y_train, X_test, y_test

    elif type == 'canny':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        # normalize inputs from 0-255 to 0.0-1.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        return X_train, y_train, X_test, y_test

    else:
        print('--- type not recognized')
        return None, None, None, None


def complex_cnn(num_classes):
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


def import_cifar10(category=0, plot=False):
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    filter_train = np.load('dataset/x_train_{}.npy'.format(categories[category]))

    if(plot):
        images = filter_train[:16]
        # create a grid of 3x3 images
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(toimage(images[i]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return filter_train



def main():
    filename = 'out/cifar100split_complex_cnn.h5'

    X_train_color, y_train_color, X_test_color, y_test_color = load_data('color')
    X_train_canny, y_train_canny, X_test_canny, y_test_canny = load_data('canny')

    # Build two models, one for color image and one for canny-edge
    model_color = complex_cnn(y_test_color.shape[1])
    model_canny = complex_cnn(y_test_color.shape[1])
    model_merge = Add()([model_color.output, model_canny.output])

    model = load(model_merge, filename)

    model = train(model, [X_train_color, X_train_canny], [y_train_color, y_train_canny], [X_test_color, X_test_canny], [y_test_color, y_test_canny])

    evaluate(model, [X_test_color, X_test_canny], [y_test_color, y_test_canny])

    save(model, filename)


if __name__ == '__main__':
    main()