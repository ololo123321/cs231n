import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N = x.shape[0]
    out = x.reshape(N, -1) @ w + b
    cache = x, w, b
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]
    dx = dout @ w.T
    dw = x.reshape(N, -1).T @ dout
    db = np.sum(dout, axis=0)
    return dx.reshape(x.shape), dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    dx = (x >= 0) * dout
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    ax = bn_param.get('ax', 0)

    D = x.shape[1]
    if len(x.shape) == 2:
        running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    else:
        running_mean = bn_param.get('running_mean', np.zeros((1, D, 1, 1), dtype=x.dtype))
        running_var = bn_param.get('running_var', np.zeros((1, D, 1, 1), dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        mu = np.mean(x, axis=ax, keepdims=True)
        var = np.var(x, axis=ax, keepdims=True)
        inv_std = (var + eps) ** -0.5
        x_centered = x - mu
        x_norm = x_centered * inv_std
        out = x_norm * gamma + beta
        cache = mu, inv_std, gamma, x_centered, x_norm, ax

        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mu
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * var
    elif mode == 'test':
        out = (x - running_mean) * (running_var + eps) ** -0.5
        out = out * gamma + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    mu, inv_std, gamma, x_centered, x_norm, ax = cache
    m = dout.shape[ax]

    dx_norm = dout * gamma
    dvar = -0.5 * np.sum(dx_norm * x_centered, axis=ax, keepdims=True) * inv_std ** 3
    dmu = -inv_std * np.sum(dx_norm, axis=ax, keepdims=True) - 2 * dvar * np.mean(x_centered, axis=ax, keepdims=True)
    dx = dx_norm * inv_std + (2 * x_centered * dvar + dmu) / m
    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dy, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    mu, inv_std, gamma, x_centered, x_norm, ax = cache

    dx = inv_std * gamma * (dy - np.mean(dy, axis=ax, keepdims=True) - inv_std * x_norm * np.mean(dy * x_centered, axis=ax, keepdims=True))
    dgamma = np.sum(x_norm * dy, axis=ax, keepdims=True)
    dbeta = np.sum(dy, axis=ax, keepdims=True)
    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get('eps', 1e-5)
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    inv_std = (var + eps) ** -0.5
    x_centered = x - mu
    x_norm = x_centered * inv_std
    out = x_norm * gamma + beta
    cache = mu, inv_std, gamma, x_centered, x_norm, 1
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    return batchnorm_backward(dout, cache)


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    cache = None
    if mode == 'train':
        mask = (np.random.randn(*x.shape) < p) / p
        out = x * mask
        cache = dropout_param, mask
    elif mode == 'test':
        out = x
    else:
        raise ValueError('invalid mode')

    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    p, s = conv_param['pad'], conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = (H - HH + 2*p) // s + 1
    W_out = (W - WW + 2*p) // s + 1
    out = np.zeros((N, F, H_out, W_out))

    if p:
        x = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)

    for i in range(N):
        for j in range(F):
            for k in range(H_out):
                for m in range(W_out):
                    img_slice = x[i, :, k*s:k*s+HH, m*s:m*s+WW]
                    out[i, j, k, m] = np.sum(img_slice * w[j]) + b[j]

    cache = x, w, b, conv_param
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    p, s = conv_param['pad'], conv_param['stride']
    N, F, H, W = dout.shape
    _, _, HH, WW = w.shape

    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(F):
            for m in range(H):
                for l in range(W):
                    dout_ijml = dout[i, j, m, l]
                    dx[i, :, m*s:m*s+HH, l*s:l*s+WW] += w[j] * dout_ijml
                    dw[j] += x[i, :, m*s:m*s+HH, l*s:l*s+WW] * dout_ijml
    db = np.sum(dout, axis=(0, 2, 3))

    if p:
        dx = dx[:, :, p:-p, p:-p]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    h, w, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    H_out = (H - h) // s + 1
    W_out = (W - w) // s + 1
    out = np.zeros((N, C, H_out, W_out))

    indices = []
    for i in range(N):
        for j in range(C):
            for k in range(H_out):
                for m in range(W_out):
                    a = x[i, j, k*s:k*s+h, m*s:m*s+w]
                    out[i, j, k, m] = np.max(a)
                    k0, m0 = np.unravel_index(a.argmax(), a.shape)
                    indices.append((i, j, k*s+k0, m*s+m0))

    cache = x.shape, indices
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x_shape, indices = cache

    dx = np.zeros(x_shape)
    dout_flatten = dout.flatten()
    for i, idx in enumerate(indices):
        dx[idx] = dout_flatten[i]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    return batchnorm_forward(x, gamma, beta, bn_param)


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    return batchnorm_backward_alt(dout, cache)


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of
    batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    eps = gn_param.get('eps', 1e-5)
    N, C, H, W = x.shape
    x = x.reshape(N, G, C//G, H, W)
    mu = np.mean(x, axis=(2, 3, 4), keepdims=True)
    var = np.var(x, axis=(2, 3, 4), keepdims=True)
    inv_std = (var + eps) ** -0.5
    x_centered = (x - mu)
    x_norm = x_centered * inv_std
    out = x_norm.reshape(N, C, H, W) * gamma + beta
    cache = inv_std, gamma, x_centered, x_norm, G
    return out, cache


def spatial_groupnorm_backward(dy, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    inv_std, gamma, x_centered, x_norm, G = cache
    N, C, H, W = dy.shape

    dx_norm = (dy * gamma).reshape(N, G, C//G, H, W)
    a = np.sum(dx_norm * x_centered, axis=(2, 3, 4), keepdims=True)
    b = np.sum(dx_norm, axis=(2, 3, 4), keepdims=True)
    c = np.mean(x_centered, axis=(2, 3, 4), keepdims=True)

    dvar = -0.5 * a * inv_std ** 3
    dmu = -inv_std * b - 2 * dvar * c
    dx = dx_norm * inv_std + (2 * x_centered * dvar + dmu) / (C//G*H*W)
    dx = dx.reshape(N, C, H, W)

    dgamma = np.sum(x_norm.reshape(N, C, H, W) * dy, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dy, axis=(0, 2, 3), keepdims=True)
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
