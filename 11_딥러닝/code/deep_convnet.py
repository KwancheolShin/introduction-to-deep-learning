# deep_convnet.py -- a deeper CNN for MNIST (Week: deep learning), numpy only
#   [Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> Pool] x 3 -> Affine(50) -> ReLU -> Dropout -> Affine(10)
#   * every filter is 3x3, He initialization, ReLU, Dropout; trained with Adam
#   * uses conv_layers.py and layers.py of the CNN week (copied to this folder)
# Run this file to check the gradients of backward() against numerical gradients.

import numpy as np
from layers import Affine, Relu, Dropout, SoftmaxWithCrossEntropy
from conv_layers import Convolution, Pooling, Flatten, output_size


class DeepConvNet:
    #             (filters, pad)  for the six 3x3 convolutions
    CONV = [(16, 1), (16, 1), (32, 1), (32, 2), (64, 1), (64, 1)]

    def __init__(self, input_shape=(1, 28, 28), hidden_size=50, output_size_=10,
                 dropout_ratio=0.5):
        C, H, _ = input_shape
        self.params = {}
        self.layers = []
        k = 0
        for i, (FN, pad) in enumerate(self.CONV):
            k += 1
            W = np.sqrt(2.0 / (C * 3 * 3)) * np.random.randn(FN, C, 3, 3)   # He
            B = np.zeros(FN)
            self.params[f"W{k}"], self.params[f"B{k}"] = W, B
            self.layers += [Convolution(W, B, pad=pad), Relu()]
            C, H = FN, output_size(H, 3, 1, pad)
            if i % 2 == 1:                             # after every second conv
                self.layers.append(Pooling(2, 2))
                H = H // 2
        flat = C * H * H                               # 64 * 4 * 4 = 1024
        for n_in, n_out in ((flat, hidden_size), (hidden_size, output_size_)):
            k += 1
            self.params[f"W{k}"] = np.sqrt(2.0 / n_in) * np.random.randn(n_in, n_out)
            self.params[f"B{k}"] = np.zeros((1, n_out))
        self.layers += [Flatten(),
                        Affine(self.params["W7"], self.params["B7"]), Relu(), Dropout(dropout_ratio),
                        Affine(self.params["W8"], self.params["B8"])]
        self.last = SoftmaxWithCrossEntropy()

    def predict(self, X, train_flg=False):    # scores U
        for layer in self.layers:
            X = layer.forward(X, train_flg) if isinstance(layer, Dropout) else layer.forward(X)
        return X

    def loss(self, X, T):                     # training: dropout on
        return self.last.forward(self.predict(X, train_flg=True), T)

    def backward(self):                       # call right after loss()
        d = self.last.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)
        grads, k = {}, 0
        for layer in self.layers:
            if isinstance(layer, (Convolution, Affine)):
                k += 1
                grads[f"W{k}"], grads[f"B{k}"] = layer.dW, layer.dB
        return grads

    def accuracy(self, X, T, batch_size=500):  # test: dropout off
        correct = 0
        for i in range(0, X.shape[0], batch_size):
            U = self.predict(X[i:i + batch_size])
            correct += np.sum(np.argmax(U, axis=1) == np.argmax(T[i:i + batch_size], axis=1))
        return correct / X.shape[0]

    def num_params(self):
        return sum(p.size for p in self.params.values())

    def load_params(self, file_name):
        """copy saved arrays INTO the existing ones (the layers keep pointing at them)"""
        saved = np.load(file_name)
        for key, p in self.params.items():
            p[...] = saved[key]

    def astype(self, dtype):
        """change every parameter to dtype (e.g. np.float16), keeping the layers connected"""
        k = 0
        for layer in self.layers:
            if isinstance(layer, (Convolution, Affine)):
                k += 1
                layer.W = self.params[f"W{k}"] = self.params[f"W{k}"].astype(dtype)
                layer.B = self.params[f"B{k}"] = self.params[f"B{k}"].astype(dtype)


if __name__ == "__main__":
    np.random.seed(0)
    net = DeepConvNet(dropout_ratio=0.0)      # no dropout: the loss must not be random
    print("number of parameters:", net.num_params())
    X = np.random.rand(2, 1, 28, 28)
    T = np.eye(10)[[3, 7]]
    net.loss(X, T)
    grads = net.backward()
    h = 1e-5
    for key in ("W1", "W4", "W6", "W7", "B8"):
        P = net.params[key]
        idx = tuple(np.random.randint(s) for s in P.shape)
        old = P[idx]
        P[idx] = old + h
        lp = net.loss(X, T)
        P[idx] = old - h
        lm = net.loss(X, T)
        P[idx] = old
        print(f"{key}{list(idx)}: backward {grads[key][idx]: .6e}   numerical {(lp - lm) / (2 * h): .6e}")
