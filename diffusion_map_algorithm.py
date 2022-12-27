import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits import mplot3d
import pandas as pd
import random
from sklearn.datasets import load_digits
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import math
import scipy


random.seed(1)



def generate_unit_circle_points(n_points, n_dimension, distribution=None, 
                                noise=False, noise_mean=0, noise_var=1):

    # Initalizing Data Matrix
    x = np.zeros((n_dimension,n_points))

    # Generate Two Random Orthonormal Vectors
    u = np.random.standard_normal(n_dimension)
    u /= np.linalg.norm(u)
    v = np.random.standard_normal(n_dimension)
    v -= v.dot(u) * u
    v /= np.linalg.norm(v)


    # Generate Points Using Different Distributions
    if distribution=="uniform":
        for i in range(n_points):
            theta = np.random.uniform(0, 2 * np.pi)
            x[:,i] = np.cos(theta) * u + np.sin(theta) * v
      
    elif distribution=="beta":
        for i in range(n_points):
            #theta = np.random.beta(2,5) * 2 * np.pi
            theta = (np.random.beta(5,2) * (2 * np.pi + 0.2) / 0.4) - 0.2

            x[:,i] = np.cos(theta) * u + np.sin(theta) * v
   
    else:
        for i in range(n_points):
            theta = i * 2 * np.pi / n_points
            x[:,i] = np.cos(theta) * u + np.sin(theta) * v
          
    # Adding Noise
    if noise:
        x += np.random.normal(noise_mean, noise_var, size=(n_dimension,n_points))

    return np.transpose(x)

def generate_unit_sphere_points(n_points, n_dimension, distribution="uniform", 
                                noise=False, noise_mean=0, noise_var=1):

    # Initializing Data Matrix
    x = np.zeros((n_dimension,n_points))


    # Generate Points Using Different Distribution
    if distribution == "uniform":
        for i in range(n_points):
            point = np.random.standard_normal(n_dimension)
            x[:,i] = point / np.linalg.norm(point)

    elif distribution == "beta":
        for i in range(n_points):
            # point = np.random.beta(5,2,n_dimension) * 2 * np.pi
            point = (np.random.beta(5,2,n_dimension) * (2 * np.pi + 0.2) / 0.4) - 0.2
            x[:,i] = point / np.linalg.norm(point)
    else:
        print("Please Select \"Uniform\" or \"Beta\" Distrubtion")
         
    #Adding Noise
    if noise:
        x += np.random.normal(noise_mean, noise_var, size=(n_dimension,n_points))

    return np.transpose(x)


def generate_rotating_img_points(img_number, n_images, padding, 
                                  noise=False, noise_mean=0, noise_var=1):

    # Loading MNIST Dataset via Sklearn
    digits = load_digits()

    # Chooisng a Particular Image to Generate Points
    img = digits.images[img_number]

    size=(n_images,(8 + 2 * padding)**2)

    # Generating Data Matrix
    x = np.zeros(size)

    # Adding Data Points for Each Rotation
    for i in range(n_images):
        theta = i * 360 / n_images
        img_padded = np.pad(img, padding)
        img_rotate = ndimage.rotate(img_padded, theta, reshape=False)
        img_flat = img_rotate.flatten()

        x[i,:] = img_flat

    if noise:
        x += np.random.normal(noise_mean, noise_var, size=size)

    return x

def kernel(x, y, epsilon):
    return np.exp(-1 * np.linalg.norm(x - y)**2 / (4 * epsilon**2))


def generate_diffusion_matrix(x, epsilon):

    # Find number of data points
    n = x.shape[0]

    # Initalize weight matrix
    W = np.zeros((n,n))

    # Fill in weight matrix
    for i in range(n):
        for j in range(n):
            W[i,j] = kernel(x[i,:], x[j,:], epsilon)

    # Normalize weight matrix
    row_sums = np.sum(W,axis=0)
    D_inverse = np.diag(1 / row_sums)
    W_normalized = np.matmul(D_inverse, W)


    # Find similar matrix A to find eigenvectors (To avoid complex eigenvalues)
    D_left = np.diag(row_sums**0.5)
    D_right = np.diag(row_sums**-0.5)
    A = np.matmul(D_left, np.matmul(W_normalized, D_right))

    return A.real, D_left

def generate_diffusion_map(A, D_left):

    # Find and sort eigenvectors based on eigenvalues (smallest to largest)
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = eigenValues.argsort()[::-1] 
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    # Just return eigenVectors???
    return np.matmul(D_left,eigenVectors)


def lle(X, dimension, method='k_nearest_neighbors', K_neighbors = 10, epsilon = 1):

    d = dimension
    K = K_neighbors

    if method == 'k_nearest_neighbors':
        k_nearest_neighbors = True
    elif method == 'epsilon_neighborhood':
        k_nearest_neighbors = False
    else:
        print("[Method Not Found]")
        return

    D, N = X.shape

    print('\nLLE running on ' + str(N) + ' points in ' + str(D) + ' dimensions\n')

    print('-->Finding ' + str(K) + ' nearest neighbors.\n')

    X2 = np.sum(X**2, axis=0)
    x = np.tile(X2, (N,1))
    y = np.transpose(x)
    xy = np.matmul(np.transpose(X),X)
    distance_squared = x + y - 2*xy
    np.fill_diagonal(distance_squared,0)

    distance = np.sqrt(distance_squared)
    
    if k_nearest_neighbors:
        neighborhood = np.transpose(np.argsort(distance)[:,1:(1+K)])

    else:
        epsilon_radius = np.multiply(distance < epsilon, distance > 0)
        a,b = np.nonzero(epsilon_radius)
  
        neighborhood = []
        for val in range(N):
            matching_val = np.where(a == val)
            if np.sum(matching_val) == 0:
                print("[Epsilon is too small. Each point needs at least one neighbor.]\n")
                return
            neighborhood.append(b[matching_val])

    print('-->Solving for reconstruction weights.\n')

    
    if k_nearest_neighbors:
        if K > D:
            print('    [note: K>D; regularization will be used]\n')
            tol=1e-3
        else:
            tol=0

        W = np.zeros((K,N))

        for ii in range(N):

            z = X[:,neighborhood[:,ii]] - np.transpose(np.tile(X[:,ii],(K,1)))
            C = np.matmul(np.transpose(z), z)
            C = C + np.eye(K)  * tol * np.trace(C)
            W[:,ii] = np.transpose(np.matmul(np.linalg.inv(C),np.ones((K,1))))
            W[:,ii] = W[:,ii] / np.sum(W[:,ii])

    else:

        lengths = [len(x) for x in neighborhood]
        W = np.zeros((max(lengths),N))

        for ii in range(N):
            l = len(neighborhood[ii])
            z = X[:,neighborhood[ii]] - np.transpose(np.tile(X[:,ii],(l,1)))
            C = np.matmul(np.transpose(z), z)
            C = C + np.eye(l)  * N * epsilon**(d + 3)
            W[:l,ii] = np.transpose(np.matmul(np.linalg.inv(C),np.ones((l,1))))


    print('-->Computing embedding.\n')
    M = np.eye(N)
    
    for ii in range(N):

        if k_nearest_neighbors:
            w = W[:,ii]
            jj = neighborhood[:,ii]

        else:
            w = W[len(neighborhood[ii]) - 1,ii]
            jj = neighborhood[ii]

        M[ii,jj] = M[ii,jj] - w
        M[jj,ii] = M[jj,ii] - w
        M[np.ix_(jj,jj)] = M[np.ix_(jj,jj)] + np.outer(w, w)


    eigenvals, Y= scipy.linalg.eigh(M, eigvals_only=False, subset_by_index=[1,d])

    Y = Y * np.sqrt(N)


    print('Done.\n')

    return Y

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



# Main Parameters
n_points=1000
n_dimension=3
# epsilon_list = [i/10 for i in range(1,11)]
epsilon_list = [0.6,0.7]

# Method Type
# Options are 'diffusion_map' or 'lle'
main_method = 'diffusion_map'


# LLE options
intrinsic_dimension = 2
# Options are 'k_nearest_neighbors' or 'epsilon_neighborhood'
lle_type = 'k_nearest_neighbors'
if lle_type == 'k_nearest_neighbors' and main_method == 'lle':
    epsilon_list == [0]
# Number of nearest neighbors to find.
K = 20

# Points Parameters
# Options are "unit_circle", "unit_sphere", or "rotating_int"
points_type = "unit_circle"
# Options are "uniform" or "beta" Distribution
distribution="uniform"

# Noise Parameters
noise=True
noise_mean=0
noise_var=0.1

# Image Parameters
img_number = 13
n_images = 750
padding = 3

# Visualization Parameters
visualization_type = "2d_comparison"
idx = [1,2,3]



for i in range(len(epsilon_list)):
    print("\n")
    print("Epsilon: " + str(epsilon_list[i]))
    print("Generating [" + points_type + "] Points...")

    if points_type == "unit_circle":
        points = generate_unit_circle_points(n_points,n_dimension, distribution=distribution,
                                              noise=noise, noise_mean=noise_mean, noise_var=noise_var)
    elif points_type == "unit_sphere":
        points = generate_unit_sphere_points(n_points,n_dimension,distribution=distribution,
                                              noise=noise,noise_mean=noise_mean,noise_var=noise_var)
    elif points_type == "rotating_int":
        points = generate_rotating_img_points(img_number, n_images, padding, noise=noise,                
                                               noise_mean=noise_mean,noise_var=noise_var)
    else:
        print("Select Points Type")
        break    


    if main_method == 'diffusion_map':
        print("Generating Diffusion Matrix...")
        A, D_left = generate_diffusion_matrix(points, epsilon_list[i])

        print("Generating Diffusion Map...")
        results = generate_diffusion_map(A, D_left)

    elif main_method == 'lle':
        results = lle(points, dimension=2, method=lle_type, K_neighbors=K, epsilon=epsilon_list[i])

    else:
        print("Method not found.")

    print("Creating " + visualization_type + " Visualization...")
    
 

    if visualization_type == "2d":
        generate_2d_plot(results, idx=idx)
    elif visualization_type == "2d_comparison":
        generate_2d_plot_comparison(results, points, idx=idx)
    elif visualization_type == "3d":
        generate_3d_plot(results,idx=idx)
    elif visualization_type == "3d_comparison":
        generate_3d_plot_comparison(results,idx=idx)
    elif visualization_type == "rotating_int":
        generate_digits_plot(results, points, idx=idx)
    else:
        print("Select Visualization Type")
        break
