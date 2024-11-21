import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils import plot_const_density, plot_decision_boundary, plot_gaussian, bayes_learn, bayes_predict
from utils import naive_bayes_learn, evaluate_all_clfs_ROC, plot_confusion_matrix, plot_eig_vectors, evaluate_all_clfs_DET

# Read the data
print('Enter absolute path of dataset (eg. E:/Documents/ML_PATH/PRML/Assignment - 2/Linear/): ')
file_path = input()

train = pd.read_csv(file_path + "train.txt", header = None)
dev = pd.read_csv(file_path + "dev.txt", header = None)

# Visualize the training data
sns.scatterplot(x = 0, y = 1, hue = 2, data = train)
plt.title('Visualization of Training data')
plt.show()

# Train-test split
dev, test = train_test_split(dev, test_size = 0.5, random_state = 0)

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_dev = dev.iloc[:, :-1]
y_dev = dev.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

classes = y_train.unique()
h = 300
# Prepare a mesh grid
x_min, x_max = X_train[0].min() - 1, X_train[0].max() + 1
y_min, y_max = X_train[1].min() - 1, X_train[1].max() + 1
n = min(x_max - x_min, y_max - y_min)/h
xx, yy = np.meshgrid(np.arange(x_min, x_max, n), np.arange(y_min, y_max, n))

print('Bayes (B) or Naive Bayes (NB) ?')
clf = input()
fun = 0
SAME_C = False
SAME_SIG = False
if clf == 'B':
    fun = bayes_learn
    print('Same covariance (A) or different covariance matrices for the classes (B) ?')
    key = input()
    if key == 'A':
        SAME_C = True
else:
    fun = naive_bayes_learn
    print('Same covariance (A) or different covariance matrices for the classes (B) ?')
    key = input()
    if key == 'A':
        print('Do you want $C = \sigma^2I$ (Yes/No) ?')
        ans = input()
        if ans == 'Yes':
            SAME_SIG = True
        else:
            SAME_C = True

# Parameter estimation
mu, sig, prior = fun(X_train, y_train, classes, same_C = SAME_C, same_sig = SAME_SIG)

# Visualize the eigen vectors of the covariance matrices
plot_eig_vectors(X_train, mu, sig, classes, h = h)

# Prediction on train and test data
train_score, train_pred = bayes_predict(X_train.to_numpy(), mu, sig, prior, classes)
dev_score, dev_pred = bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)

# Confusion matrices on train and test data
plot_confusion_matrix(y_train, train_pred, classes, title = 'Train Set')
plot_confusion_matrix(y_dev, dev_pred, classes)

# 3D plot of the gaussian likelihood functions
plot_gaussian([xx, yy], mu, sig, classes)

# Visualization of the constant density curves
plot_const_density(X_train, y_train, mu, sig, classes, h = h)
plot_const_density(X_dev, y_dev, mu, sig, classes, title = 'Development Set', h = h)

# Visualization of the decision boundaries
plot_decision_boundary(X_train, y_train, mu, sig, prior, classes, h = h)
plot_decision_boundary(X_dev, y_dev, mu, sig, prior, classes, title = 'Development Set', h = h) 

# ROC and DET curves
print('Compare all the classifiers (Yes/No) ?')
key = input()
if key == 'Yes':
    scores = []
    mu, sig, prior = bayes_learn(X_train, y_train, classes)
    scores.append(bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)[0])
    mu, sig, prior = bayes_learn(X_train, y_train, classes, same_C = True)
    scores.append(bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)[0])
    mu, sig, prior = naive_bayes_learn(X_train, y_train, classes)
    scores.append(bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)[0])
    mu, sig, prior = naive_bayes_learn(X_train, y_train, classes, same_C = True)
    scores.append(bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)[0])
    mu, sig, prior = naive_bayes_learn(X_train, y_train, classes, same_sig = True)
    scores.append(bayes_predict(X_dev.to_numpy(), mu, sig, prior, classes)[0])
    evaluate_all_clfs_ROC(y_dev, scores)
    evaluate_all_clfs_DET(y_dev, scores)