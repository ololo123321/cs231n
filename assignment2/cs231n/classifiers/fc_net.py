import numpy as np

from cs231n.tensorflow_toy.ops import *
from cs231n.tensorflow_toy.base import *
from cs231n.tensorflow_toy.utils import *
from cs231n.tensorflow_toy.gradients import *

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet:
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W0'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b0'] = np.zeros(hidden_dim)
        self.params['W1'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        out, cache0 = affine_relu_forward(X, self.params['W0'], self.params['b0'])
        scores, cache1 = affine_forward(out, self.params['W1'], self.params['b1'])

        if y is None:
            return scores

        loss, dy = softmax_loss(scores, y)
        for name, param in self.params.items():
            if 'W' in name:
                loss += 0.5 * self.reg * np.sum(param * param)

        grads = {}
        dy, dw1, db1 = affine_backward(dy, cache1)
        dy, dw0, db0 = affine_relu_backward(dy, cache0)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1
        grads['W0'] = dw0 + self.reg * self.params['W0']
        grads['b0'] = db0
        return loss, grads


class FullyConnectedNet:
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An :nteger giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = len(hidden_dims)
        self.dtype = dtype
        self.num_classes = num_classes
        self.params = {}

        for i in range(self.num_layers + 1):
            if i == 0:
                h0, h1 = input_dim,  hidden_dims[0]
            elif i == self.num_layers:
                h0, h1 = hidden_dims[-1], num_classes
            else:
                h0, h1 = hidden_dims[i-1], hidden_dims[i]
            self.params[f'W{i}'] = np.random.randn(h0, h1) * weight_scale
            self.params[f'b{i}'] = np.zeros(h1)

            if self.normalization and i != self.num_layers:  # перед выходным слоем нормализация не нужна
                self.params[f'gamma{i}'] = np.ones(h1)
                self.params[f'beta{i}'] = np.zeros(h1)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_params = {}
        if self.use_dropout:
            self.dropout_params = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_params['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.normalization_params = []
        if self.normalization == 'batchnorm':
            self.normalization_params = [{'mode': 'train'} for _ in range(self.num_layers)]
        if self.normalization == 'layernorm':
            self.normalization_params = [{} for _ in range(self.num_layers)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        affine_cache = {}
        normalization_cache = {}
        relu_cache = {}
        dropout_cache = {}
        out = X
        scores = None

        # FORWARD
        for i in range(self.num_layers+1):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            if i != self.num_layers:
                # affine
                out, cache = affine_forward(out, W, b)
                affine_cache[i] = cache
                # normalization
                if self.normalization:
                    gamma, beta, params = self.params[f'gamma{i}'], self.params[f'beta{i}'], self.normalization_params[i]
                    if self.normalization == 'batchnorm':
                        out, cache = batchnorm_forward(out, gamma, beta, params)
                    else:
                        out, cache = layernorm_forward(out, gamma, beta, params)
                    normalization_cache[i] = cache
                # relu
                out, cache = relu_forward(out)
                relu_cache[i] = cache
                # dropout
                if self.use_dropout:
                    out, cache = dropout_forward(out, self.dropout_params)
                    dropout_cache[i] = cache
            else:
                scores, cache = affine_forward(out, W, b)
                affine_cache[i] = cache

        if y is None:
            return scores

        # LOSS
        loss, dx = softmax_loss(scores, y)
        for name, param in self.params.items():
            if 'W' in name:
                loss += 0.5 * self.reg * np.sum(param * param)

        # BACKWARD
        grads = {}
        for i in range(self.num_layers, -1, -1):
            if i == self.num_layers:
                dx, dw, db = affine_backward(dx, affine_cache[i])
            else:
                # dropout
                if self.use_dropout:
                    dx = dropout_backward(dx, dropout_cache[i])
                # relu
                dx = relu_backward(dx, relu_cache[i])
                # normalization
                if self.normalization:
                    params = normalization_cache[i]
                    if self.normalization == 'batchnorm':
                        dx, dgamma, dbeta = batchnorm_backward_alt(dx, params)
                    else:
                        dx, dgamma, dbeta = layernorm_backward(dx, params)
                    grads[f'gamma{i}'] = dgamma
                    grads[f'beta{i}'] = dbeta
                # affine
                dx, dw, db = affine_backward(dx, affine_cache[i])
            grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
            grads[f'b{i}'] = db

        return loss, grads


# class FullyConnectedNet:
#     """
#     A fully-connected neural network with an arbitrary number of hidden layers,
#     ReLU nonlinearities, and a softmax loss function. This will also implement
#     dropout and batch/layer normalization as options. For a network with L layers,
#     the architecture will be
#
#     {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
#
#     where batch/layer normalization and dropout are optional, and the {...} block is
#     repeated L - 1 times.
#
#     Similar to the TwoLayerNet above, learnable parameters are stored in the
#     self.params dictionary and will be learned using the Solver class.
#     """
#
#     def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
#                  dropout=1, normalization=None, reg=0.0,
#                  weight_scale=1e-2, dtype=np.float32, seed=None):
#         """
#         Initialize a new FullyConnectedNet.
#
#         Inputs:
#         - hidden_dims: A list of integers giving the size of each hidden layer.
#         - input_dim: An :nteger giving the size of the input.
#         - num_classes: An integer giving the number of classes to classify.
#         - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
#           the network should not use dropout at all.
#         - normalization: What type of normalization the network should use. Valid values
#           are "batchnorm", "layernorm", or None for no normalization (the default).
#         - reg: Scalar giving L2 regularization strength.
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - dtype: A numpy datatype object; all computations will be performed using
#           this datatype. float32 is faster but less accurate, so you should use
#           float64 for numeric gradient checking.
#         - seed: If not None, then pass this random seed to the dropout layers. This
#           will make the dropout layers deteriminstic so we can gradient check the
#           model.
#         """
#         self.normalization = normalization
#         self.use_dropout = dropout != 1
#         self.reg = reg
#         self.num_layers = len(hidden_dims)
#         self.dtype = dtype
#         self.num_classes = num_classes
#         self.params = {}
#
#         self.dims = [input_dim] + hidden_dims + [num_classes]
#         n = len(self.dims)
#         for i in range(n - 1):
#             h0, h1 = self.dims[i], self.dims[i + 1]
#             self.params[f'W{i}'] = np.random.randn(h0, h1) * weight_scale
#             self.params[f'b{i}'] = np.zeros(h1)
#             if self.normalization == 'batchnorm' and i != n - 2:  # перед выходным слоем дропаут не нужен
#                 self.params[f'gamma{i}'] = np.ones(h1)
#                 self.params[f'beta{i}'] = np.zeros(h1)
#
#         # When using dropout we need to pass a dropout_param dictionary to each
#         # dropout layer so that the layer knows the dropout probability and the mode
#         # (train / test). You can pass the same dropout_param to each dropout layer.
#         self.dropout_param = {}
#         if self.use_dropout:
#             self.dropout_param = {'mode': 'train', 'p': dropout}
#             if seed is not None:
#                 self.dropout_param['seed'] = seed
#
#         # With batch normalization we need to keep track of running means and
#         # variances, so we need to pass a special bn_param object to each batch
#         # normalization layer. You should pass self.bn_params[0] to the forward pass
#         # of the first batch normalization layer, self.bn_params[1] to the forward
#         # pass of the second batch normalization layer, etc.
#         self.bn_params = []
#         if self.normalization == 'batchnorm':
#             self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers)]
#         if self.normalization == 'layernorm':
#             self.bn_params = [{} for _ in range(self.num_layers)]
#
#         for k, v in self.params.items():
#             self.params[k] = v.astype(dtype)
#
#     def loss(self, X_value, y_value=None):
#         """
#         Compute loss and gradient for the fully-connected net.
#
#         Input / output: Same as TwoLayerNet above.
#         """
#         mode = None
#         if self.normalization == 'batchnorm':
#             mode = 'test' if y_value is None else 'train'
#         n = len(self.dims)
#         loss, grads, layer = None, None, None
#         loss_reg = 0
#         X = Placeholder()
#         y = Placeholder()
#         for i in range(n - 1):
#             W_value = self.params[f'W{i}']
#             W = Variable(W_value, name=f'W{i}')
#             b = Variable(self.params[f'b{i}'], name=f'b{i}')
#             gamma = Variable(self.params.get(f'gamma{i}'), name=f'gamma{i}')
#             beta = Variable(self.params.get(f'beta{i}'), name=f'beta{i}')
#             loss_reg += 0.5 * self.reg * np.sum(W_value * W_value)
#             if i < n-2:
#                 fc = dense(X, W, b) if i == 0 else dense(layer, W, b)
#                 fc = batchnorm(fc, gamma, beta, mode) if self.normalization == 'batchnorm' else fc
#                 layer = relu(fc)
#             else:
#                 scores = dense(layer, W, b)
#                 if y_value is None:
#                     return run(scores, {X: X_value})
#                 loss = neg(reduce_sum(multiply(y, log(softmax(scores)))))
#
#         feed_dict = {X: X_value, y: np.eye(self.num_classes)[y_value]}
#         loss_value = run(loss, feed_dict)
#         loss_value += loss_reg
#         grads = gradients(loss, feed_dict)
#
#         grads_values = {}
#         for node, grad in grads.items():
#             if node.name:
#                 grads_values[node.name] = grad
#                 if 'W' in node.name:
#                     grads_values[node.name] += self.reg * self.params[node.name]
#         return loss_value, grads_values
