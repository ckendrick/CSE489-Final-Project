import os
import os.path as path

from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D

import matplotlib.pyplot as plt


class AutoEncoder_model():

    # build the model
    def __init__(self, model_name="model", encoding_dimension=32, input_dimension=784, output_dimension=784):
        self.model_name = model_name

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
        # autoencoder = Model(input_img, decoded))
        # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        #
        # # this model maps an input to its encoded representation
        # encoder = Model(input_img, encoded)
        #
        # return encoder, autoencoder

    def checkpoint(self):
        if not path.exists('weights'):
            os.mkdir('weights')
        # records checkpoints per EPOCH using keras
        print('--- enabling ModelCheckpoint')
        check_path = 'weights/{}_weights.best.hdf5'.format(self.model_name)
        return ModelCheckpoint(check_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # train the model
    def train(self, x_train, x_test, epoch_step=10, batch_size=256, shuffle=True):
        # train the model
        self.model.fit(x_train, x_train,
                       epochs=epoch_step,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_data=(x_test, x_test),
                       callbacks=[self.checkpoint()])

    def plot_output(self, x, filename=None):
        # encode the images
        encoded_imgs = self.encoder.predict(x)
        # decode the images
        decoded_imgs = self.decoder.predict(encoded_imgs)

        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display encoded
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(encoded_imgs[i].reshape(16, 2))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n*2)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()