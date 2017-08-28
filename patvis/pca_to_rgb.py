# utilities for taking a high-dimensional matrix and producing rgb color values from its rows. 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def squeeze(v):
    # translate and scale a vector onto the interval [0,1]
    if min(v) < 0:
        v -= min(v)
    return v/np.linalg.norm(v)


def low_d(Z):
    # suppose Z is some high-dimensional matrix. 
    # project it onto 3 dimensions.
    pca = PCA(n_components=3)
    return pca.fit_transform(Z)


def main(n_vectors=100, vector_dim=1000):
    Z = np.random.random(size=(n_vectors, vector_dim))
    xs,ys = np.random.random(size=(2,n_vectors))
    z = np.array(list(map(squeeze, low_d(Z)))) 
    #squeeze each vector into the interval [0,1]. 
    plt.scatter(xs,ys,facecolors=z)
    plt.show()

    
