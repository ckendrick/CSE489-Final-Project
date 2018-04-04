import numpy as np

import matplotlib.pyplot as plt
from scipy.misc import toimage

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

for i in range(10):
    import_cifar10(i, True)