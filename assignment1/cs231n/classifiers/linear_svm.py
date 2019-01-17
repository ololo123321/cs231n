import numpy as np


def svm_loss_naive(W, X, y, reg=0):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (C, D) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros_like(W)  # (num_classes, num_features)
    n_classes = W.shape[0]
    n_train = X.shape[0]
    delta = 1
    loss = 0
    for i in range(n_train):
        xi, yi = X[i], y[i]
        scores = W @ xi  # (num_classes, 1)
        si = scores[yi]
        for j in range(n_classes):
            if j == yi:
                continue
            sj = scores[j]
            margin = sj - si + delta
            loss += max(0, margin)
            grad = (margin > 0) * xi  # grad xi w.r.t wj
            dW[yi] -= grad  # grad xi w.r.t wyi
            dW[j] += grad
    loss /= n_train
    dW /= n_train
    loss += reg * np.sum(np.square(W))
    dW += 2 * reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg=0):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    n_train = X.shape[0]
    delta = 1
    target_locs = y, np.arange(0, n_train)

    scores = W @ X.T  # (n_classes, n_train)
    s = scores[target_locs]
    margins = scores - s + delta
    margins[target_locs] = 0
    mask = margins > 0
    loss = 1 / n_train * np.sum(margins[mask]) + reg * np.sum(np.square(W))

    S = mask.astype(float)
    S[target_locs] = -1 * np.sum(S, axis=0)
    dW = 1 / n_train * S @ X + 2 * reg * W
    return loss, dW
