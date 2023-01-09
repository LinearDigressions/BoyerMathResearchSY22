import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits import mplot3d
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def generate_3d_plot_comparison(x, y, idx=[0,1,2]):
    fig = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,[idx[0]]], x[:,idx[1]],x[:,idx[2]])
    ax.scatter3D(y[:,0], y[:,1],y[:,2])
    plt.show()


def generate_3d_plot(x,idx=[0,1,2]):
    fig = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,idx[0]], x[:,idx[1]],x[:,idx[2]])
    plt.show()


def generate_2d_plot(x, idx=[0,1]):
    fig,ax = plt.subplots()

    ax.scatter(x[:,idx[0]], x[:,idx[1]])
    plt.show()


def generate_2d_plot_comparison(x, y, idx=[0,1]):
    fig,ax = plt.subplots()
    ax.scatter(x[:,idx[0]], x[:,idx[1]])
    ax.scatter(y[:,0], y[:,1])
    plt.show()


def generate_digits_plot(x, images, idx=[0,1]):

    fig,ax = plt.subplots()

    shown_images = np.array([[1.0, 1.0]])
    X = np.array([x[:,idx[0]], x[:,idx[1]]]).T
    X = MinMaxScaler().fit_transform(X)

    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits

        dist = np.sum((X[i] - shown_images) ** 2, 1)

        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        img = images[i].reshape(14,14)
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img), X[i])
        ax.add_artist(imagebox)
    
    plt.show()

