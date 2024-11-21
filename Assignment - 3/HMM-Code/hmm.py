import os
import numpy as np

def train_hmm(dir, states, symb, sum_p = 0.01, seed = 0):
    os.system('make clean')
    os.system('make all')
    comm = "./train_hmm '{}' {} {} {} {}".format(dir, seed, states, symb, sum_p)
    os.system(comm)

def test_hmm(dir, model):
    os.system('make clean')
    os.system('make all')
    comm = "./test_hmm '{}' '{}'".format(dir, model)
    os.system(comm)
    curr_dir = os.getcwd() + '/alphaout'
    return np.loadtxt(curr_dir)

def predict_hmm(dir, hmm):
    scores = []
    for model in hmm:
        scores.append(test_hmm(dir, model))
    return np.array(scores).T