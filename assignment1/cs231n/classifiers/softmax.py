import numpy as np


def softmax_loss_naive(W, X, y, reg=0):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    N = X.shape[0]
    C = W.shape[1]
    loss = 0
    dW = np.zeros_like(W)
    for i in range(N):
        xi, yi = X[i], y[i]
        scores = W.T @ xi
        si = scores[yi]
        scores_exp_sum = np.sum(np.exp(scores))
        loss += -si + np.log(scores_exp_sum)
        for j in range(C):
            dW[:, j] += xi * np.exp(scores[j]) / scores_exp_sum
        dW[:, yi] -= xi
    loss /= N
    loss += reg * np.sum(W * W)
    dW /= N
    dW += 2 * reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg=0):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    N - число объектов
    D - размерность
    C - число классов
    """
    N = X.shape[0]
    scores = X @ W  # N x C
    scores_exp = np.exp(scores)
    probs = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    target_logprobs = np.log(probs[range(N), y_dev])
    data_loss = -np.sum(target_logprobs) / N
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss

    probs[range(N), y] -= 1
    dW = X.T @ probs / N + 2 * reg * W
    return loss, dW
