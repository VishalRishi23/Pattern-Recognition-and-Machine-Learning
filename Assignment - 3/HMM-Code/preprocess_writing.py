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
        arr = np.loadtxt(dir + file)
        nf = arr[0]
        arr_res = np.reshape(arr[1:], (int(nf), nc))
        seq = get_sequence(arr_res, centroids)
        s_vector.append(seq)
    return s_vector

def get_features(dir, nc):
    f_vector = []
    files = os.listdir(dir)
    for file in files:
        arr = np.loadtxt(dir + file)
        nf = arr[0]
        arr_res = np.reshape(arr[1:], (int(nf), nc))
        f_vector.append(arr_res)
    return np.concatenate(f_vector, axis = 0)

def write_seq(dir, arr):
    with open(dir, 'w') as fp:
        for a in arr:
            fp.write(' '.join([str(k) for k in a]) + '\n')