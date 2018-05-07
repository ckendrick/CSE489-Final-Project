import os
import os.path as path

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt


class AutoEncoder_model():

    # build the model
    def __init__(self, model_name="model", encoding_dimension=32, input_dimension=784, output_dimension=784):
        self.model_name = model_name
        self.net_weights = []

        input_img = Input(shape=(input_dimension,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dimension, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(output_dimension, activation='sigmoid')(encoded)

        ################################################################################################################
        # autoencoder model
        self.model = Model(input_img, decoded)

        ################################################################################################################
        # model for encoding images
        self.encoder = Model(input_img, encoded)

        ################################################################################################################
        # model for decoding images

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dimension,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.model.layers[-1]
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

        ################################################################################################################
        # compile the network
        self.model.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model.summary()

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