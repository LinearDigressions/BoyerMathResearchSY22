import numpy as np
import scipy
import numpy as np
import pandas as pd


def lle_epsilon_neighbors(X, dimension, epsilon = 1):

    d = dimension
 
    D, N = X.shape

    print('\nLLE running on ' + str(N) + ' points in ' + str(D) + ' dimensions\n')

    print('-->Finding ' + str(epsilon) + ' epsilon neighbors.\n')
    
    X2 = np.sum(X**2, axis=0)
    x = np.tile(X2, (N,1))
    y = np.transpose(x)
    xy = np.matmul(np.transpose(X),X)
    distance_squared = x + y - 2*xy
    np.fill_diagonal(distance_squared,0)

    distance = np.sqrt(distance_squared)

    
  
    epsilon_radius = np.multiply(distance < epsilon, distance > 0)

    a,b = np.nonzero(epsilon_radius)
    
    neighborhood = []
    for val in range(N):
        matching_val = np.where(a == val)

        if np.size(matching_val) == 0:
            neighborhood.append([])
            print("Epsilon Neighborhood Too Small for a Point")
            continue
            # print("[Epsilon is too small. Each point needs at least one neighbor.]\n")
            # print(X[0,val])
            # print(X[1,val])
            # return
            
        neighborhood.append(b[matching_val])
    
    print('-->Solving for reconstruction weights.\n')

    lengths = [len(x) for x in neighborhood]
    print("Mean Neighbors: " + str(sum(lengths)/len(lengths)))
    W = np.zeros((N,N))
    

    for ii in range(N):
        l = len(neighborhood[ii])

        if l == 0:
            continue

        z = X[:,neighborhood[ii]] - np.transpose(np.tile(X[:,ii],(l,1)))
        C = np.matmul(np.transpose(z), z)
        C = C + np.eye(l) * N * epsilon**(d + 3)

        W[neighborhood[ii],ii] = np.transpose(np.matmul(np.linalg.inv(C),np.ones((l,1))))

        W[neighborhood[ii],ii] = W[neighborhood[ii],ii] / np.sum(W[neighborhood[ii],ii])
    
   
            

    print('-->Computing embedding.\n')
    M = np.eye(N)
    
    for ii in range(N):



        w = W[neighborhood[ii],ii]
        jj = neighborhood[ii]

        M[ii,jj] = M[ii,jj] - w
        M[jj,ii] = M[jj,ii] - w
        M[np.ix_(jj,jj)] = M[np.ix_(jj,jj)] + np.outer(w, w)


    eigenvals, Y= scipy.linalg.eigh(M, eigvals_only=False, subset_by_index=[1,d+2])

    Y = Y * np.sqrt(N)


    print('Done.\n')

    return Y


def lle_nearest_neighbors(X, dimension, K_neighbors = 10):

    d = dimension
    K = K_neighbors

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
    
    neighborhood = np.transpose(np.argsort(distance)[:,1:(1+K)])

    print('-->Solving for reconstruction weights.\n')

    
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

   

    print('-->Computing embedding.\n')
    M = np.eye(N)
    
    for ii in range(N):

        w = W[:,ii]
        jj = neighborhood[:,ii]

        M[ii,jj] = M[ii,jj] - w
        M[jj,ii] = M[jj,ii] - w
        M[np.ix_(jj,jj)] = M[np.ix_(jj,jj)] + np.outer(w, w)


    eigenvals, Y= scipy.linalg.eigh(M, eigvals_only=False, subset_by_index=[1,d+2])

    Y = Y * np.sqrt(N)

    print('Done.\n')

    return Y



