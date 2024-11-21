import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, DetCurveDisplay
from kmeans import k_means
from preprocess_writing import get_features, get_symbols
from dtw import classify_DTW
from classification import compute_DET, plot_confusion_matrix, compute_ROC

# Parameters
NC = 2 # Dimension of cluster means
classes = [0, 1, 2, 3, 4]
n_clusters = 20 # No. of clusters
y = pd.Series([0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20) # Targets for dev dataset

print('Absolute path of training data of class "a": ')
a_train_dir = input() #'Hand writing data/a/train/'
print('Absolute path of training data of class "ai": ')
ai_train_dir = input() #'Hand writing data/ai/train/'
print('Absolute path of training data of class "ba": ')
ba_train_dir = input() #'Hand writing data/ba/train/'
print('Absolute path of training data of class "da": ')
da_train_dir = input() #'Hand writing data/da/train/'
print('Absolute path of training data of class "la": ')
la_train_dir = input() #'Hand writing data/la/train/'
train_dir = [a_train_dir, ai_train_dir, ba_train_dir, da_train_dir, la_train_dir]

print('Absolute path of dev data of class "a": ')
a_dev_dir = input() #'Hand writing data/a/dev/'
print('Absolute path of dev data of class "ai": ')
ai_dev_dir = input() #'Hand writing data/ai/dev/'
print('Absolute path of dev data of class "ba": ')
ba_dev_dir = input() #'Hand writing data/ba/dev/'
print('Absolute path of dev data of class "da": ')
da_dev_dir = input() #'Hand writing data/da/dev/'
print('Absolute path of dev data of class "la": ')
la_dev_dir = input() #'Hand writing data/la/dev/'

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
train = [a_s_vector, ai_s_vector, ba_s_vector, da_s_vector, la_s_vector]

dev_a = get_symbols(a_dev_dir, NC, centroid)
dev_ai = get_symbols(ai_dev_dir, NC, centroid)
dev_ba = get_symbols(ba_dev_dir, NC, centroid)
dev_da = get_symbols(da_dev_dir, NC, centroid)
dev_la = get_symbols(la_dev_dir, NC, centroid)

# DTW
print('DTW...')
pred_a, pred_target_a = classify_DTW(dev_a, train)
pred_ai, pred_target_ai = classify_DTW(dev_ai, train)
pred_ba, pred_target_ba = classify_DTW(dev_ba, train)
pred_da, pred_target_da = classify_DTW(dev_da, train)
pred_la, pred_target_la = classify_DTW(dev_la, train)

# Classification performance
print('Classification performance: ')
scores = np.concatenate([pred_a, pred_ai, pred_ba, pred_da, pred_la], axis = 0)
y_pred = np.concatenate([pred_target_a, pred_target_ai, pred_target_ba, pred_target_da, pred_target_la], axis = 0)
tpr, fpr_1, _ = compute_ROC(y, scores)
fpr, fnr, _ = compute_DET(y, scores)

plot_confusion_matrix(y, y_pred, classes = classes)

plt.plot(fpr_1, tpr, linewidth = 2)
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