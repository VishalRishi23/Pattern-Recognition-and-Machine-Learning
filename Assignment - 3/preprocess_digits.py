import os
import numpy as np

def get_sequence(arr, centroids):
    n = arr.shape[0]
    seq = []
    for i in range(n):
        cluster = np.argmin(np.sum((centroids - arr[i, :])**2, axis = 1))
        seq.append(cluster)
    return seq

def get_symbols(dir, nc, centroids):
    s_vector = []
    files = os.listdir(dir)
    for file in files:
        if '.mfcc' in file:
            arr = np.loadtxt(dir + file, skiprows = 1)
            # nf = arr[0]
            # arr_res = np.reshape(arr, (int(nf), nc))
            seq = get_sequence(arr, centroids)
            s_vector.append(seq)
    return s_vector

def get_features(dir, nc):
    f_vector = []
    files = os.listdir(dir)
    for file in files:
        if '.mfcc' in file:
            arr = np.loadtxt(dir + file, skiprows = 1)
            # nf = arr[0]
            # arr_res = np.reshape(arr, (int(nf), nc))
            f_vector.append(arr)
    return np.concatenate(f_vector, axis = 0)