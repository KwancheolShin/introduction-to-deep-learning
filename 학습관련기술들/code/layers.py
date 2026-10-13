# layers.py -- layers and the multi-layer network (from mnist_lab.py / mnist_mlp.py)
#   Affine, Relu, Sigmoid, softmax, SoftmaxWithCrossEntropy, SoftmaxWithSquaredError
#   BatchNormalization, Dropout
#   MultiLayerNet: X -> [Affine -> (BatchNorm) -> activation -> (Dropout)] x k -> Affine -> Softmax-loss
#                  (+ weight decay)

import numpy as np


class Affine:
    """A = X W + B  for a mini-batch X of shape (h, l)."""
    def __init__(self, W, B):
        self.W = W          # (l, m)
        self.B = B          # (1, m)
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.B          # (h,l)@(l,m) + (1,m) -> (h, m)

    def backward(self, dA):
        self.dW = self.X.T @ dA                          # (l,h)@(h,m) -> (l, m)
        self.dB = np.sum(dA, axis=0, keepdims=True)      # sum over rows -> (1, m)
        return dA @ self.W.T                             # dL/dX : (h, l)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, A):
        self.mask = (A > 0)
        return np.where(self.mask, A, 0.0)

    def backward(self, dZ):
        return np.where(self.mask, dZ, 0.0)


class Sigmoid:
    def __init__(self):
        self.Z = None

    def forward(self, A):
        self.Z = 1.0 / (1.0 + np.exp(-A))
        return self.Z

    def backward(self, dZ):
        return dZ * self.Z * (1.0 - self.Z)


def softmax(U):
    U = U - np.max(U, axis=1, keepdims=True)
    expU = np.exp(U)
    return expU / np.sum(expU, axis=1, keepdims=True)


class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, U, T):
        self.T = T
        self.Y = softmax(U)
        h = U.shape[0]
        return -np.sum(T * np.log(self.Y + 1e-12)) / h

    def backward(self):
        h = self.T.shape[0]
        return (self.Y - self.T) / h        # the ONLY place where we divide by h


class SoftmaxWithSquaredError:
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, U, T):
        self.T = T
        self.Y = softmax(U)
        h = U.shape[0]
        return 0.5 * np.sum((self.Y - self.T) ** 2) / h

    def backward(self):
        h = self.T.shape[0]
        R = self.Y - self.T
        return self.Y * (R - np.sum(self.Y * R, axis=1, keepdims=True)) / h


class BatchNormalization:
    """Normalize each node (column) over the mini-batch, then scale and shift:
       x_hat = (x - mu) / sqrt(var + eps),   y = gamma * x_hat + beta"""
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-7):
        self.gamma = gamma          # (1, m)
        self.beta = beta            # (1, m)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = None    # used at test time
        self.running_var = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, X, train_flg=True):
        if self.running_mean is None:
            self.running_mean = np.zeros((1, X.shape[1]))
            self.running_var = np.ones((1, X.shape[1]))
        if train_flg:                                     # statistics of THIS mini-batch
            mu = np.mean(X, axis=0, keepdims=True)        # (1, m)
            xc = X - mu                                   # (h, m)
            var = np.mean(xc ** 2, axis=0, keepdims=True) # (1, m)
            std = np.sqrt(var + self.eps)
            xn = xc / std                                 # x_hat (h, m)
            self.h, self.xc, self.std, self.xn = X.shape[0], xc, std, xn
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:                                             # test time: running averages
            xn = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * xn + self.beta

    def backward(self, dY):
        self.dbeta = np.sum(dY, axis=0, keepdims=True)              # (1, m)
        self.dgamma = np.sum(self.xn * dY, axis=0, keepdims=True)   # (1, m)
        dxn = self.gamma * dY                                       # (h, m)
        dxc = dxn / self.std
        dstd = -np.sum(dxn * self.xc / (self.std * self.std), axis=0, keepdims=True)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.h) * self.xc * dvar
        dmu = np.sum(dxc, axis=0, keepdims=True)
        return dxc - dmu / self.h                                   # dL/dX (h, m)


class Dropout:
    """Inverted dropout (the same as PyTorch nn.Dropout)
       training: erase each node w.p. dropout_ratio, scale the survivors by 1/keep
       test:     use every node as it is"""
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, X, train_flg=True):
        if train_flg:
            keep = 1.0 - self.dropout_ratio
            self.mask = (np.random.rand(*X.shape) < keep) / keep   # 0 or 1/keep
            return X * self.mask
        return X

    def backward(self, dY):
        return dY * self.mask              # erased nodes pass no gradient


class MultiLayerNet:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="relu", loss="ce", weight_init_std=0.05, use_batchnorm=False,
                 weight_decay_lambda=0.0, use_dropout=False, dropout_ratio=0.5):
        sizes = [input_size] + list(hidden_sizes) + [output_size]   # e.g. [784, 256, 128, 10]
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}
        self.layers = []
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i + 1]
            if weight_init_std == "xavier":            # Var(W) = 1/n_in  (sigmoid, tanh)
                std = np.sqrt(1.0 / n_in)
            elif weight_init_std == "he":              # Var(W) = 2/n_in  (ReLU)
                std = np.sqrt(2.0 / n_in)
            else:                                      # a fixed number, e.g. 0.01
                std = weight_init_std
            W = std * np.random.randn(n_in, n_out)     # (input nodes, output nodes)
            B = np.zeros((1, n_out))
            self.params[f"W{i + 1}"] = W
            self.params[f"B{i + 1}"] = B
            self.layers.append(Affine(W, B))
            if i < len(sizes) - 2:                     # hidden layer
                if use_batchnorm:                      # Affine -> BatchNorm -> activation
                    gamma, beta = np.ones((1, n_out)), np.zeros((1, n_out))
                    self.params[f"gamma{i + 1}"] = gamma
                    self.params[f"beta{i + 1}"] = beta
                    self.layers.append(BatchNormalization(gamma, beta))
                self.layers.append(Relu() if activation == "relu" else Sigmoid())
                if use_dropout:                        # ... -> activation -> Dropout
                    self.layers.append(Dropout(dropout_ratio))
        self.last = SoftmaxWithCrossEntropy() if loss == "ce" else SoftmaxWithSquaredError()

    def predict(self, X, train_flg=False):    # scores U
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):   # training and test differ
                X = layer.forward(X, train_flg)
            else:
                X = layer.forward(X)
        return X

    def loss(self, X, T):                     # used for training: batch statistics, dropout
        data_loss = self.last.forward(self.predict(X, train_flg=True), T)
        # weight decay: + (lambda/2) * sum of W^2 over all weight matrices (not B, gamma, beta)
        sum_W2 = sum(np.sum(W ** 2) for key, W in self.params.items() if key.startswith("W"))
        return data_loss + 0.5 * self.weight_decay_lambda * sum_W2

    def backward(self):                       # call right after loss()
        d = self.last.backward()
        for layer in reversed(self.layers):   # the same rule, layer after layer
            d = layer.backward(d)
        grads, k = {}, 0
        for layer in self.layers:
            if isinstance(layer, Affine):
                k += 1
                grads[f"W{k}"], grads[f"B{k}"] = layer.dW, layer.dB
            elif isinstance(layer, BatchNormalization):
                grads[f"gamma{k}"], grads[f"beta{k}"] = layer.dgamma, layer.dbeta
        for key in grads:                     # weight decay: gradient + lambda * W
            if key.startswith("W"):
                grads[key] = grads[key] + self.weight_decay_lambda * self.params[key]
        return grads

    def accuracy(self, X, T):                 # used for evaluation: running statistics
        U = self.predict(X)
        return np.mean(np.argmax(U, axis=1) == np.argmax(T, axis=1))

    def num_params(self):
        return sum(p.size for p in self.params.values())
