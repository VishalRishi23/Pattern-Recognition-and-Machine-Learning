import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, DetCurveDisplay

def bayes_learn(X, y, classes, same_C = False, same_sig = False):
    c = len(classes)
    mu, sig, prior = [], [], []
    for i in range(c):
        C = X[y == classes[i]].to_numpy()
        prior.append(C.shape[0]/X.shape[0])
        mu.append(np.expand_dims(np.mean(C, axis = 0), axis = 1))
        if same_C:
            sig.append(np.dot((C - mu[-1].T).T, C - mu[-1].T)*prior[i]/C.shape[0])
        else:    
            sig.append(np.dot((C - mu[-1].T).T, C - mu[-1].T)/C.shape[0])
    if same_C:
        sig = [sum(sig)]*c
    return mu, sig, prior

def naive_bayes_learn(X, y, classes, same_C = False, same_sig = False):
    c = len(classes)
    mu, sig, prior = [], [], []
    for i in range(c):
        C = X[y == classes[i]].to_numpy()
        prior.append(C.shape[0]/X.shape[0])
        mu.append(np.expand_dims(np.mean(C, axis = 0), axis = 1))
        S = np.diag(np.diag(np.dot((C - mu[-1].T).T, C - mu[-1].T))/C.shape[0])
        if same_C:
            sig.append(S*prior[i])
        elif same_sig:
            sig.append(np.diag([np.mean(np.diag(S))]*X.shape[-1])*prior[i])
        else:    
            sig.append(S)
    if same_C or same_sig:
        sig = [sum(sig)]*c
    return mu, sig, prior

def get_params(mu, sig, prior):
    sig_inv = np.linalg.inv(sig)
    W = -0.5*sig_inv
    w = np.dot(sig_inv, mu)
    w0 = -0.5*(np.dot(mu.T, np.dot(sig_inv, mu)) + np.log(np.linalg.det(sig))) + np.log(prior)
    return [W, w, w0]

def disc_fun(x, mu, sig, prior):
    params = get_params(mu, sig,  prior)
    W, w, w0 = params
    score = np.dot(x.T, np.dot(W, x)) + np.dot(w.T, x) + w0
    return params, score

def bayes_predict(X, mu, sig, prior, classes):
    n = X.shape[0]
    c = len(classes)
    scores = []
    pred = []
    for i in range(n):
        x = X[i]
        score_x = []
        for j in range(c):
            _, score = disc_fun(x, mu[j], sig[j], prior[j])
            score_x.append(score)
        scores.append(score_x)
        pred.append(classes[np.argmax(score_x)])
    return np.exp(np.array(scores))[:, :, 0, 0], pred

def gauss_pdf(X, mu, sig, norm = False):
    f = []
    sig_inv = np.linalg.inv(sig)
    for i in range(X.shape[0]):
        x = np.expand_dims(X[i], axis = 1)
        a = np.dot((x - mu).T, np.dot(sig_inv, x - mu))
        if norm:
            a = np.exp(-0.5*a)
        f.append(a)
    return f
    
def plot_decision_boundary(X, y, mu, sig, prior, classes, colors = 'brg', h = 500, title = 'Train Set'):
    # create a mesh to plot in
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    n = min(x_max - x_min, y_max - y_min)/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, n), np.arange(y_min, y_max, n))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    _, Z = bayes_predict(np.c_[xx.ravel(), yy.ravel()], mu, sig, prior, classes)
    Z = np.array(Z)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis("tight")

    # Plot also the training points
    for i, color in zip(classes, colors):
        idx = np.where(y == i)
        plt.scatter(
            X.to_numpy()[idx, 0],
            X.to_numpy()[idx, 1],
            c = color,
            label = classes[i - 1],
            cmap = plt.cm.Paired,
            edgecolor = "black",
            s = 20,
        )
    plt.title("Decision surface of multi-class Bayes classifier - " + title)
    plt.axis("tight")
    plt.legend()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def plot_gaussian(mesh, mu, sig, classes, colors = ['red', 'blue', 'green']):
    c = len(classes)
    xx, yy = mesh
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection ='3d')
    for i in range(c):
        Z = gauss_pdf(np.c_[xx.ravel(), yy.ravel()], mu[i], sig[i], norm = True)
        Z = np.array(Z)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        cs = ax.plot_wireframe(xx, yy, Z, colors = colors[i])
    ax.legend(classes)
    ax.set_title("Class conditional Gaussian density functions")
    plt.show()

def plot_const_density(X, y, mu, sig, classes, h = 500, colors = ['red', 'blue', 'green'], title = 'Train Set'):
    c = len(classes)
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    n = min(x_max - x_min, y_max - y_min)/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, n), np.arange(y_min, y_max, n))
    for i in range(c):
        Z = gauss_pdf(np.c_[xx.ravel(), yy.ravel()], mu[i], sig[i])
        Z = np.array(Z)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        cs = plt.contour(xx, yy, Z, levels = range(1, 10), colors = colors[i], linestyles = 'dashed')
    plt.axis("tight")

    for i, color in zip(classes, colors):
        idx = np.where(y == i)
        plt.scatter(
            X.to_numpy()[idx, 0],
            X.to_numpy()[idx, 1],
            c = color,
            label = classes[i - 1],
            cmap = plt.cm.Paired,
            edgecolor = "black",
            s = 20,
        )
    plt.title("Constant density curves of multi-class Bayes classifier - " + title)
    plt.axis("tight")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def evaluate_classifier(y, scores, thresh = 0.5):
    enc = OneHotEncoder()
    y_sparse = enc.fit_transform(y.to_numpy().reshape(-1, 1)).toarray()
    prob = scores/np.sum(scores, axis = -1)[:, np.newaxis]
    n, c = prob.shape
    cm = np.zeros((2, 2))
    for i in range(c):
        col = []
        for j in range(y.shape[0]):
            if prob[j, i] >= thresh:
                col.append(1)
            else:
                col.append(0)
        cm += confusion_matrix(y_sparse[:, i], col)
    tpr = cm[1, 1]/(cm[1, 1] + cm[1, 0])
    fpr = cm[0, 1]/(cm[0, 1] + cm[0, 0])
    return tpr, fpr, cm

def compute_ROC(y, scores, h = 0.05):
    threshold = [i*h for i in range(int(1/h + 1))]
    tpr, fpr = [], []
    for thresh in threshold:
        a, b, _ = evaluate_classifier(y, scores, thresh = thresh)
        tpr.append(a)
        fpr.append(b)
    return tpr, fpr, threshold

def compute_DET(y, scores, h = 0.05):
    threshold = [i*h for i in range(int(1/h + 1))]
    fpr, fnr = [], []
    for thresh in threshold:
        _, b, cm = evaluate_classifier(y, scores, thresh = thresh)
        fpr.append(b)
        fnr.append(cm[1, 0]/(cm[1, 0] + cm[1, 1]))
    return fpr, fnr, threshold

def evaluate_all_clfs_ROC(y, score_clfs, labels = ['Bayes - different C', 'Bayes - same C', 'Naive Bayes - different C', 'Naive Bayes - same C', 'Naive Bayes - $C=\sigma^2I$']):
    for i in range(len(score_clfs)):
        tpr, fpr, _ = compute_ROC(y, score_clfs[i])
        plt.plot(fpr, tpr, linewidth = 2, label = labels[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR - Recall')
    plt.title('Receiver Operator Characteristics (ROC)')
    plt.legend()
    plt.show()

def evaluate_all_clfs_DET(y, score_clfs, labels = ['Bayes - different C', 'Bayes - same C', 'Naive Bayes - different C', 'Naive Bayes - same C', 'Naive Bayes - $C=\sigma^2I$']):
    fig, axs = plt.subplots()
    for i in range(len(score_clfs)):
        fpr, fnr, _ = compute_DET(y, score_clfs[i])
        DetCurveDisplay(fpr = fpr, fnr = fnr, estimator_name = labels[i]).plot(ax = axs)
    plt.title('Detection Error Tradeoff curves (DET)')
    plt.show()

def plot_confusion_matrix(y, pred, classes, title = 'Development Set'):
    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(5, 4))
    ax= plt.subplot()
    sns.heatmap(confusion_matrix(y, pred), annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(classes, fontsize = 10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(classes, fontsize = 10)
    plt.yticks(rotation=0)

    plt.title('Confusion Matrix - ' + title, fontsize=20)
    plt.show()

def plot_eig_vectors(X, mu, sig, classes, h = 500, t = 10, colors = ['red', 'blue', 'green']):
    c = len(classes)
    x_min, x_max = X[0].min() - 1, X[0].max() + 1
    y_min, y_max = X[1].min() - 1, X[1].max() + 1
    n = min(x_max - x_min, y_max - y_min)/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, n), np.arange(y_min, y_max, n))
    for i in range(c):
        Z = gauss_pdf(np.c_[xx.ravel(), yy.ravel()], mu[i], sig[i])
        Z = np.array(Z)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        cs = plt.contour(xx, yy, Z, levels = range(1, 3), colors = colors[i], linestyles = 'dashed')
        _, v = np.linalg.eig(sig[i])
        for j in range(v.shape[0]):
            a = [mu[i][0], mu[i][0] + v[0, j]*n*t]
            b = [mu[i][-1], mu[i][-1] + v[-1, j]*n*t]
            plt.plot(a, b, colors[i][0] + '--')
    plt.axis("tight")
    plt.title("Constant density curves of multi-class Bayes classifier")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(classes)
    plt.show()