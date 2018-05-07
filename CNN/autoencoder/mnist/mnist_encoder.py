########################################################################################################################
## Build the model:

from autoencoder_model import AutoEncoder_model
model = AutoEncoder_model(model_name="mnist_encoder")

########################################################################################################################
## Build the dataset:

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

########################################################################################################################
## Train and sleep the model:

e_step = 10
e_max = 100
for e in range(0, e_max, e_step):
    print('--- training on {}/{} epochs'.format(e, e_max))

    model.plot_output(x_test, "{}_{}e.png".format(model.model_name, e))

    model.train(x_train, x_test, epoch_step=e_step)


print('--- done')