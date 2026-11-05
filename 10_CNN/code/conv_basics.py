# Week: CNN -- convolution and pooling, step by step
#   (1) the examples of the lecture notes: convolution, bias, padding, stride, max pooling
#   (2) im2col: the same convolution with loops and with ONE matrix product (and the speed)
#   (3) gradient check of SimpleConvNet (Convolution, Pooling, Flatten, Affine)
#   * uses conv_layers.py and layers.py in the same folder

import time
import numpy as np

from conv_layers import im2col, Convolution, Pooling, SimpleConvNet

# =====================================================
# (1) One channel, one filter: slide the filter, multiply elementwise, add
# =====================================================
def conv2d_naive(x, f, stride=1, pad=0):
    """2-D convolution of one image x (H, W) with one filter f (FH, FW), with loops"""
    x = np.pad(x, pad)                                  # zeros around the image
    FH, FW = f.shape
    OH = (x.shape[0] - FH) // stride + 1
    OW = (x.shape[1] - FW) // stride + 1
    y = np.zeros((OH, OW))
    for i in range(OH):
        for j in range(OW):
            window = x[i * stride:i * stride + FH, j * stride:j * stride + FW]
            y[i, j] = np.sum(window * f)                # multiply elementwise, then add
    return y


x = np.array([[1, 2, 3, 0],
              [0, 1, 2, 3],
              [3, 0, 1, 2],
              [2, 3, 0, 1]], dtype=float)
f = np.array([[2, 0, 1],
              [0, 1, 2],
              [1, 0, 2]], dtype=float)
print("convolution (4x4 input, 3x3 filter):\n", conv2d_naive(x, f))
print("with bias 3:\n", conv2d_naive(x, f) + 3)
print("padding 1:\n", conv2d_naive(x, f, pad=1))

x7 = np.arange(49, dtype=float).reshape(7, 7)
print("7x7 input, 3x3 filter, stride 1 -> shape", conv2d_naive(x7, f, stride=1).shape)
print("7x7 input, 3x3 filter, stride 2 -> shape", conv2d_naive(x7, f, stride=2).shape)

pool = Pooling(2, 2)                                    # max pooling, 2x2 window, stride 2
xp = np.array([[1, 2, 1, 0],
               [0, 1, 2, 3],
               [3, 0, 1, 2],
               [2, 4, 0, 1]], dtype=float)
print("2x2 max pooling:\n", pool.forward(xp.reshape(1, 1, 4, 4))[0, 0])

# =====================================================
# (2) im2col: every filter position becomes one ROW
# =====================================================
x1 = np.random.rand(1, 3, 7, 7)                         # 1 image, 3 channels, 7x7
print("im2col of (1, 3, 7, 7) with a 5x5 filter:", im2col(x1, 5, 5).shape)
x2 = np.random.rand(10, 3, 7, 7)                        # 10 images
print("im2col of (10, 3, 7, 7) with a 5x5 filter:", im2col(x2, 5, 5).shape)


def conv_loops(X, W, B):
    """(N, C, H, W) with FN filters (FN, C, FH, FW): every image, every filter, every position"""
    N, C, H, Wd = X.shape
    FN, _, FH, FW = W.shape
    OH, OW = H - FH + 1, Wd - FW + 1
    Y = np.zeros((N, FN, OH, OW))
    for n in range(N):
        for k in range(FN):
            for i in range(OH):
                for j in range(OW):
                    Y[n, k, i, j] = np.sum(X[n, :, i:i + FH, j:j + FW] * W[k]) + B[k]
    return Y


np.random.seed(0)
X = np.random.rand(100, 1, 28, 28)                      # a mini-batch of MNIST size
W = np.random.randn(30, 1, 5, 5)                        # 30 filters of 5x5
B = np.random.randn(30)

t0 = time.time()
Y_loops = conv_loops(X, W, B)
t_loops = time.time() - t0
t0 = time.time()
Y_im2col = Convolution(W, B).forward(X)
t_im2col = time.time() - t0
print(f"output shape {Y_im2col.shape}, max |difference| = {np.max(np.abs(Y_loops - Y_im2col)):.1e}")
print(f"loops {t_loops:.2f} s,  im2col {t_im2col:.3f} s  ({t_loops / t_im2col:.0f} times faster)")

# =====================================================
# (3) Gradient check: backprop vs numerical differentiation on a small SimpleConvNet
# =====================================================
def numerical_gradient(f, P, eps=1e-5):
    grad = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        f_plus = f()
        P[idx] = old - eps
        f_minus = f()
        P[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


np.random.seed(1)
small = SimpleConvNet(input_shape=(2, 8, 8), filter_num=3, filter_size=3, hidden_size=5, output_size=4)
X_small = np.random.randn(5, 2, 8, 8)                   # 5 images, 2 channels, 8x8
T_small = np.eye(4)[np.random.randint(0, 4, size=5)]
small.loss(X_small, T_small)
grads = small.backward()
for key in small.params:
    num = numerical_gradient(lambda: small.loss(X_small, T_small), small.params[key])
    print(f"{key}: shape {str(small.params[key].shape):15s} "
          f"max |backprop - numerical| = {np.max(np.abs(grads[key] - num)):.2e}")
