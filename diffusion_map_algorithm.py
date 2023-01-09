import numpy as np

# Defining Kernel
def kernel(x, y, epsilon):
    return np.exp(-1 * np.linalg.norm(x - y)**2 / (4 * epsilon**2))


def generate_diffusion_matrix(x, epsilon):
    x = np.transpose(x)

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

    return np.matmul(D_left,eigenVectors)