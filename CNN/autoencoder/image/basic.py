# Simple CNN model for CIFAR-10
#
# NN-Team-2

import os
import os.path as path

from scipy import ndimage
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
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


def build_model():
    input_img = Input(shape=(3, 200, 200))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    Model(input_img, encoded).summary()

    # # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    #
    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2D(16, (3, 3), activation='relu')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #
    # autoencoder = load(Model(input_img, decoded), 'autoenc_conv')
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    #
    # # this model maps an input to its encoded representation
    # encoder = Model(input_img, encoded)
    #
    # return encoder, autoencoder

# tensorboard --logdir=run0:logs_c10_0un1:logs_c10_1,run2:logs_c10_2,run3:logs_c10_3,run4:logs_c10_4,run5:logs_c10_5,run6:logs_c10_6,run2:logs_c10_7 --port 6006
def tensorboard(log_dir):
    # starting tensorboard: tensorboard --logdir=run1:logs1/,run2:logs2/ --port 6006
    if not path.exists(log_dir):
        os.mkdir(log_dir)
    print('--- enabling TensorBoard')
    return TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)


def train(model, X_train, X_test, log_dir, epochs):
    # Fit the model
    if log_dir is None:
        callbacks = []
    else:
        callbacks = [tensorboard(log_dir)]
    model.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    callbacks=callbacks)

    return model


def evaluate(encoder, auto_encoder, x_test):
    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = auto_encoder.predict(x_test)

    import matplotlib.pyplot as plt

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    n = 10
    plt.figure(figsize=(20, 8))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


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

    X_train, _, X_test, _ = load_data()

    epochs = 50

    encoder, auto_encoder = build_model()

    # auto_encoder = train(auto_encoder, X_train, X_test, 'logs_rmsIM', epochs)
    #
    # evaluate(encoder, auto_encoder, X_test)
    #
    # save(auto_encoder, 'out/auto_encoder.h5')
    # save(encoder, 'out/encoder.h5')


if __name__ == '__main__':
    main()