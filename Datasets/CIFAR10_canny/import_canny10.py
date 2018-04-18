import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import toimage
from scipy.misc import imsave

# NOTE: the canny images have to be multiplied by 255, they are currently saved as binary (0 or 1)
def import_cifar10(category=0, plot=False, save=False):
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    filter_train = np.load('dataset/x_train_canny_{}.npy'.format(categories[category]))
    filter_train = np.reshape(filter_train, (filter_train.shape[0], 32, 32))

    images = filter_train[:16]

    # create a single image
    if(save):
        print('saving {}.png'.format(categories[category]))
        imsave('{}.png'.format(categories[category]), images[0])

    if(plot):
        # create a grid of 3x3 images
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(toimage(images[i]*255), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return filter_train

for i in range(10):
    import_cifar10(category=i, plot=True, save=False)