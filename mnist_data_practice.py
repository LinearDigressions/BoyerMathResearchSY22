from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

digits = load_digits()

img = digits.images[13]



def generate_img(img_number, n_images, padding):
    digits = load_digits()

    img = digits.images[img_number]

    X = np.zeros((n_images,(8 + 2 * padding)**2))
    for i in range(n_images):
        theta = i * 360 / n_images
        print(theta)
        img_padded = np.pad(img, padding)
        img_rotate = ndimage.rotate(img_padded, theta, reshape=False)
        # plt.matshow(img_rotate)
        # plt.show()
        img_flat = img_rotate.flatten()

        X[i,:] = img_flat

    return np.transpose(X)

X = generate_img(13, 2000, 1)
print(X.shape)