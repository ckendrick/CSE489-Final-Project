import os
import os.path as path

from keras.layers import Input, Reshape
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Deconv2D

import matplotlib.pyplot as plt


class AutoEncoder_model():

    # build the model
    def __init__(self):
        self.auto_weights = 'out/autoencoder_weights.best.hdf5'
        self.enc_weights = 'out/encoder_weights.best.hdf5'

        input_img = Input(shape=(200, 200, 3))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (1, 1), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((1, 1), padding='same')(x)

        x = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
        # x = UpSampling2D((1, 1))(x)
        x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (1, 1), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = self.load(Model(input_img, decoded), self.auto_weights)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.summary()

        # this model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

    def checkpoint(self):
        if not path.exists('weights'):
            os.mkdir('weights')
        # records checkpoints per EPOCH using keras
        print('--- enabling ModelCheckpoint')
        check_path = 'weights/{}_weights.best.hdf5'.format(self.model_name)
        return ModelCheckpoint(check_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

    # train the model
    def train(self, x_train, x_test, epoch_step=10, batch_size=128, shuffle=True):
        # train the model
        self.autoencoder.fit(x_train, x_train,
                       epochs=epoch_step,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       validation_data=(x_test, x_test))

    def plot_output(self, x, filename=None):
        # encode and decode some digits
        # note that we take them from the *test* set
        encoded_imgs = self.encoder.predict(x)
        decoded_imgs = self.autoencoder.predict(x)

        import matplotlib.pyplot as plt

        n = 10  # how many images we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x[i].reshape(200, 200, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display encoded
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(encoded_imgs[i].reshape(25, 25))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n * 2)
            plt.imshow(decoded_imgs[i].reshape(200, 200, 3))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def load(self, model, filename):
        if not path.exists('out'):
            os.mkdir('out')

        if path.exists(filename):
            model.load_weights(filename)

        return model

    def save(self):
        if not path.exists('out'):
            os.mkdir('out')

        self.autoencoder.save_weights(self.auto_weights)
        self.encoder.save_weights(self.enc_weights)