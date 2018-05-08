########################################################################################################################
## Build the model:

from autoencoder_model import AutoEncoder_model
model = AutoEncoder_model()

########################################################################################################################
## Build the dataset:

import os

from scipy import ndimage
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils import np_utils

def load_data():

    (X_train, y_train), (X_test, y_test) = import_images('../../../Datasets/Images/export')

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
                            X.append(img)
                            y.append(k)
                            # print('{}: {}/{}'.format(k, i, files.__len__()))

                k += 1

    print(np.array(X).shape)
    print(np.array(y).shape)

    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test))


X_train, _, X_test, _ = load_data()

########################################################################################################################
## Train and sleep the model:

e_step = 100
e_max = 1000
e_offset = 200
for e in range(0, e_max, e_step):
    print('--- training on {}/{} epochs'.format(e, e_max))

    model.train(X_train, X_test, epoch_step=e_step)

    model.plot_output(X_train, "train_{}e.png".format(e+e_step+e_offset))
    model.plot_output(X_test, "test_{}e.png".format(e+e_step+e_offset))

    model.save()