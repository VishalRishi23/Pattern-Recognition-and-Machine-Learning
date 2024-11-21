import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, DetCurveDisplay
from preprocess_digits import get_features, get_symbols, write_seq
from hmm import train_hmm, predict_hmm
from kmeans import k_means
from classification import compute_DET, compute_ROC, plot_confusion_matrix

# Parameters
NC = 38 # Dimension of cluster means
classes = [0, 1, 2, 3, 4]
y = pd.Series([0]*12 + [1]*12 + [2]*12 + [3]*12 + [4]*12) # Targets for dev set
n_clusters = 20 # No. of clusters - for VQ
STATES = 15 # No. States in the HMM
SYMBOLS = n_clusters # No.of Symbols in the HMM
SEED = 42

print("Absolute path to the training dataset of class '1': ")
a_train_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/1/train/'
print("Absolute path to the training dataset of class '2': ")
ai_train_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/2/train/'
print("Absolute path to the training dataset of class '4': ")
ba_train_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/4/train/'
print("Absolute path to the training dataset of class '9': ")
da_train_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/9/train/'
print("Absolute path to the training dataset of class 'o': ")
la_train_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/o/train/'
train_dir = [a_train_dir, ai_train_dir, ba_train_dir, da_train_dir, la_train_dir]

print('Output path for training data of class "1": ')
dir_a = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/1.seq'
print('Output path for training data of class "2": ')
dir_ai = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/2.seq'
print('Output path for training data of class "4": ')
dir_ba = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/4.seq'
print('Output path for training data of class "9": ')
dir_da = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/9.seq'
print('Output path for training data of class "o": ')
dir_la = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/o.seq'

print("Absolute path to the dev dataset of class '1': ")
a_val_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/1/dev/'
print("Absolute path to the dev dataset of class '2': ")
ai_val_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/2/dev/'
print("Absolute path to the dev dataset of class '4': ")
ba_val_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/4/dev/'
print("Absolute path to the dev dataset of class '9': ")
da_val_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/9/dev/'
print("Absolute path to the dev dataset of class 'o': ")
la_val_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/Isolated digits/o/dev/'

print('Output path for all the dev data: ')
dev_dir = input() #'./Documents/ML_PATH/PRML/Assignment - 3/HMM-Code/Isolated digits/dev.seq'

features = []
for dir in train_dir:
    f_vector = get_features(dir, NC)
    features.append(f_vector)
features = np.concatenate(features, axis = 0)

# Vector Quantization
print('Vector Quantization....')
clusters, centroid, distortion = k_means(features, n_clusters, max_iter = 20, thresh = 0.01)

a_s_vector = get_symbols(a_train_dir, NC, centroid)
ai_s_vector = get_symbols(ai_train_dir, NC, centroid)
ba_s_vector = get_symbols(ba_train_dir, NC, centroid)
da_s_vector = get_symbols(da_train_dir, NC, centroid)
la_s_vector = get_symbols(la_train_dir, NC, centroid)

write_seq(dir_a, a_s_vector)
write_seq(dir_ai, ai_s_vector)
write_seq(dir_ba, ba_s_vector)
write_seq(dir_da, da_s_vector)
write_seq(dir_la, la_s_vector)

# HMM Training
print('HMM Training....')
train_hmm(dir_a, STATES, SYMBOLS, seed = SEED)
train_hmm(dir_ai, STATES, SYMBOLS, seed = SEED)
train_hmm(dir_ba, STATES, SYMBOLS, seed = SEED)
train_hmm(dir_da, STATES, SYMBOLS, seed = SEED)
train_hmm(dir_la, STATES, SYMBOLS, seed = SEED)
HMM_model = [dir + '.hmm' for dir in [dir_a, dir_ai, dir_ba, dir_da, dir_la]]

dev_a = get_symbols(a_val_dir, NC, centroid)
dev_ai = get_symbols(ai_val_dir, NC, centroid)
dev_ba = get_symbols(ba_val_dir, NC, centroid)
dev_da = get_symbols(da_val_dir, NC, centroid)
dev_la = get_symbols(la_val_dir, NC, centroid)

dev_data = []
for dev in [dev_a, dev_ai, dev_ba, dev_da, dev_la]:
    for seq in dev:
        dev_data.append(seq)

write_seq(dev_dir, dev_data)

# Classification Performance
print('Classification performance')
scores = predict_hmm(dev_dir, HMM_model)
y_pred = np.argmax(scores, axis = 1)

tpr, fpr, _ = compute_ROC(y, scores)
fpr, fnr, _ = compute_DET(y, scores)

plot_confusion_matrix(y, y_pred, classes = classes)

plt.plot(fpr, tpr, linewidth = 2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR - Recall')
plt.title('Receiver Operator Characteristics (ROC)')
plt.legend()
plt.show()

fig, axs = plt.subplots()
DetCurveDisplay(fpr = fpr, fnr = fnr).plot(ax = axs)
plt.title('Detection Error Tradeoff curves (DET)')
plt.show()