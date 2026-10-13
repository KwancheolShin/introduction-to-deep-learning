# conv_layers.py -- convolution and pooling layers (Week: CNN)
#   im2col, col2im, Convolution, Pooling, Flatten, SimpleConvNet
#   Images are 4-D arrays of shape (N, C, H, W): N images, C channels, height H, width W.
#   Affine, Relu and SoftmaxWithCrossEntropy are the layers of the previous weeks (layers.py).

import numpy as np
from layers import Affine, Relu, SoftmaxWithCrossEntropy


def output_size(H, FH, stride=1, pad=0):
    """height (or width) of the output: OH = (H + 2P - FH) / S + 1"""
    return (H + 2 * pad - FH) // stride + 1


def im2col(X, FH, FW, stride=1, pad=0):
    """(N, C, H, W) -> (N*OH*OW, C*FH*FW): ONE ROW for every place the filter visits"""
    N, C, H, W = X.shape
    OH, OW = output_size(H, FH, stride, pad), output_size(W, FW, stride, pad)
    Xp = np.pad(X, [(0, 0), (0, 0), (pad, pad), (pad, pad)])    # zero padding
    col = np.zeros((N, C, FH, FW, OH, OW))
    for i in range(FH):                   # loop over the FILTER, not the image
        for j in range(FW):
            col[:, :, i, j, :, :] = Xp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride]
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)


def col2im(col, X_shape, FH, FW, stride=1, pad=0):
    """the reverse of im2col: ADD every row back where it came from"""
    N, C, H, W = X_shape
    OH, OW = output_size(H, FH, stride, pad), output_size(W, FW, stride, pad)
    col = col.reshape(N, OH, OW, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)
    Xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(FH):
        for j in range(FW):
            Xp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += col[:, :, i, j, :, :]
    return Xp[:, :, pad:pad + H, pad:pad + W]     # remove the padding


class Convolution:
    """Convolution of (N, C, H, W) with FN filters (FN, C, FH, FW) -> (N, FN, OH, OW).
       With im2col: ONE matrix product, i.e. an Affine layer at every filter position."""
    def __init__(self, W, B, stride=1, pad=0):
        self.W = W                  # (FN, C, FH, FW)
        self.B = B                  # (FN,): one bias per filter
        self.stride = stride
        self.pad = pad
        self.dW = None
        self.dB = None

    def forward(self, X):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = X.shape
        OH = output_size(H, FH, self.stride, self.pad)
        OW = output_size(W, FW, self.stride, self.pad)
        self.X_shape = X.shape
        self.col = im2col(X, FH, FW, self.stride, self.pad)     # (N*OH*OW, C*FH*FW)
        self.W_col = self.W.reshape(FN, -1).T                    # one filter per COLUMN
        A = self.col @ self.W_col + self.B                       # (N*OH*OW, FN)
        return A.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)    # (N, FN, OH, OW)

    def backward(self, dA):
        FN, C, FH, FW = self.W.shape
        dA = dA.transpose(0, 2, 3, 1).reshape(-1, FN)            # (N*OH*OW, FN)
        self.dB = np.sum(dA, axis=0)                             # the Affine rules
        self.dW = (self.col.T @ dA).T.reshape(FN, C, FH, FW)
        dcol = dA @ self.W_col.T                                 # (N*OH*OW, C*FH*FW)
        return col2im(dcol, self.X_shape, FH, FW, self.stride, self.pad)


class Pooling:
    """Max pooling: the largest value in every PH x PW window, channel by channel"""
    def __init__(self, PH, PW, stride=None, pad=0):
        self.PH = PH
        self.PW = PW
        self.stride = PH if stride is None else stride    # usually stride = window size
        self.pad = pad

    def forward(self, X):
        N, C, H, W = X.shape
        OH = output_size(H, self.PH, self.stride, self.pad)
        OW = output_size(W, self.PW, self.stride, self.pad)
        col = im2col(X, self.PH, self.PW, self.stride, self.pad)
        col = col.reshape(-1, self.PH * self.PW)           # one window per row
        self.argmax = np.argmax(col, axis=1)               # where the maximum was
        self.X_shape = X.shape
        Y = np.max(col, axis=1)
        return Y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)   # (N, C, OH, OW)

    def backward(self, dY):
        dY = dY.transpose(0, 2, 3, 1).ravel()               # the same order as col
        dcol = np.zeros((dY.size, self.PH * self.PW))
        dcol[np.arange(dY.size), self.argmax] = dY          # only the max gets it
        dcol = dcol.reshape(-1, self.X_shape[1] * self.PH * self.PW)
        return col2im(dcol, self.X_shape, self.PH, self.PW, self.stride, self.pad)


class Flatten:
    """(N, C, H, W) -> (N, C*H*W) before an Affine layer (nn.Flatten in PyTorch)"""
    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dY):
        return dY.reshape(self.X_shape)


class SimpleConvNet:
    """Conv -> ReLU -> Pool -> Flatten -> Affine -> ReLU -> Affine -> Softmax"""
    def __init__(self, input_shape=(1, 28, 28), filter_num=30, filter_size=5,
                 hidden_size=100, output_size=10):
        C, H, W = input_shape
        conv_out = H - filter_size + 1                     # 28 - 5 + 1 = 24
        pool_out = conv_out // 2                           # 2x2 max pooling: 12
        flat = filter_num * pool_out * pool_out            # 30 * 12 * 12 = 4320
        n_in = C * filter_size * filter_size               # 25 inputs per filter
        self.params = {                                    # He initialization
            "W1": np.sqrt(2.0 / n_in) * np.random.randn(filter_num, C, filter_size, filter_size),
            "B1": np.zeros(filter_num),
            "W2": np.sqrt(2.0 / flat) * np.random.randn(flat, hidden_size),
            "B2": np.zeros((1, hidden_size)),
            "W3": np.sqrt(2.0 / hidden_size) * np.random.randn(hidden_size, output_size),
            "B3": np.zeros((1, output_size)),
        }
        self.layers = [
            Convolution(self.params["W1"], self.params["B1"]),
            Relu(),
            Pooling(2, 2),
            Flatten(),
            Affine(self.params["W2"], self.params["B2"]),
            Relu(),
            Affine(self.params["W3"], self.params["B3"]),
        ]
        self.last = SoftmaxWithCrossEntropy()

    def predict(self, X):                     # scores U
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, X, T):
        return self.last.forward(self.predict(X), T)

    def backward(self):                       # call right after loss()
        d = self.last.backward()
        for layer in reversed(self.layers):   # layer after layer
            d = layer.backward(d)
        grads, k = {}, 0
        for layer in self.layers:
            if isinstance(layer, (Convolution, Affine)):
                k += 1
                grads[f"W{k}"], grads[f"B{k}"] = layer.dW, layer.dB
        return grads

    def accuracy(self, X, T, batch_size=1000):   # in pieces: im2col of 10000 images is large
        correct = 0
        for i in range(0, X.shape[0], batch_size):
            U = self.predict(X[i:i + batch_size])
            correct += np.sum(np.argmax(U, axis=1) == np.argmax(T[i:i + batch_size], axis=1))
        return correct / X.shape[0]

    def num_params(self):
        return sum(p.size for p in self.params.values())
