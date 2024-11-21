import numpy as np

def DTW(s1, s2):
    n = len(s1)
    m = len(s2)
    matrix = [[np.inf]*(m + 1) for _ in range(n + 1)]
    matrix[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            matrix[i][j] = cost + min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1])
    return matrix[n][m]

def get_score(s, X):
    n = len(X)
    scores = []
    for i in range(n):
        scores.append(DTW(s, X[i]))
    return np.mean(scores)
    
def classify_DTW(X, train):
    n = len(X)
    c = len(train)
    pred = np.zeros((n, c))
    for i in range(n):
        scores = []
        for j in range(c):
            score = get_score(X[i], train[j])
            scores.append(score)
        pred[i, :] = scores
    pred_target = np.argmin(pred, axis = 1)
    return pred, pred_target