import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits import mplot3d
import pandas as pd
import random
from sklearn.datasets import load_digits
from scipy import ndimage




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

    return x

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

    return x

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

    return np.transpose(x)

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

    if k_nearest_neighbors:
        print('-->Finding ' + str(K) + ' nearest neighbors.\n')
    else:
        print('-->Finding ' + str(epsilon) + ' epsilon neighbors.\n')
    
    X2 = np.sum(X**2, axis=0)
    x = np.tile(X2, (N,1))
    y = np.transpose(x)
    xy = np.matmul(np.transpose(X),X)
    distance_squared = x + y - 2*xy
    np.fill_diagonal(distance_squared,0)

    distance = distance_squared
    distance = np.sqrt(distance_squared)
    
    # print(distance)
    if k_nearest_neighbors:
        neighborhood = np.transpose(np.argsort(distance)[:,1:(1+K)])
    else:
        epsilon_radius = np.multiply(distance < epsilon, distance > 0)

        a,b = np.nonzero(epsilon_radius)
       
        neighborhood = []
        for val in range(N):
            matching_val = np.where(a == val)
            # print(np.size(matching_val))
            if np.size(matching_val) == 0:
                print("[Epsilon is too small. Each point needs at least one neighbor.]\n")
                print(X[0,val])
                print(X[1,val])
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
            C = C + np.eye(K) * tol * np.trace(C)
            W[:,ii] = np.transpose(np.matmul(np.linalg.inv(C),np.ones((K,1))))
            W[:,ii] = W[:,ii] / np.sum(W[:,ii])

    else:

        lengths = [len(x) for x in neighborhood]
        print("Mean Neighbors: " + str(sum(lengths)/len(lengths)))
        # print(lengths)
        W = np.zeros((max(lengths),N))
        # print(lengths)
        # print(W)
        for ii in range(N):
            l = len(neighborhood[ii])
    
            z = X[:,neighborhood[ii]] - np.transpose(np.tile(X[:,ii],(l,1)))
            C = np.matmul(np.transpose(z), z)
            C = C + np.eye(l) * N * epsilon**(d + 3)

            W[:l,ii] = np.transpose(np.matmul(np.linalg.inv(C),np.ones((l,1))))

            W[:l,ii] = W[:l,ii] / np.sum(W[:l,ii])
            
        

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


    eigenvals, Y= scipy.linalg.eigh(M, eigvals_only=False, subset_by_index=[1,d+2])

    Y = Y * np.sqrt(N)


    print('Done.\n')

    return np.transpose(Y)


np.random.seed(1)

points = generate_unit_circle_points(4000,2)

# points = generate_unit_circle_points(1000,2)
#points = generate_unit_sphere_points(1000,3, distribution="beta")
#points = generate_rotating_img_points(13, 750, 3)
# Y = lle(points, dimension=3, method='k_nearest_neighbors', epsilon=.02, K_neighbors=2)
Y = lle(points, dimension=2, method='epsilon_neighborhood', epsilon=.002, K_neighbors=16)
# fig,ax = plt.subplots()
# ax.scatter(points[0,:], points[1,:])
# plt.show()
fig,ax = plt.subplots()
ax.scatter(points[0,:], points[1,:])
plt.show()
# YOU ALREADY CUT OUT FIRST EIGENVECTOR
fig,ax = plt.subplots()
ax.scatter(Y[0,:], Y[1,:])
plt.show()


# for i in [1/1000 for i in range(5,10)]:
#     Y = lle(points, dimension=2, method='epsilon_neighborhood', epsilon=i, K_neighbors=11)
#     fig,ax = plt.subplots()

#     # YOU ALREADY CUT OUT FIRST EIGENVECTOR
#     ax.scatter(Y[:,0], Y[:,1])
#     plt.show()


