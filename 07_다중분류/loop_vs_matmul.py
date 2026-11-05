# Week: Multi-class Classification -- why mini-batches? (vectorization)
# Forward pass of one Affine layer for h = 100 samples:
#   (1) a Python for loop over the samples, one row at a time
#   (2) one matrix product  X @ W + B
import time
import numpy as np

np.random.seed(0)
h, l, m = 100, 784, 100
X = np.random.rand(h, l)
W = 0.05 * np.random.randn(l, m)
B = np.zeros((1, m))

def forward_loop(X, W, B):
    A = np.zeros((X.shape[0], W.shape[1]))
    for k in range(X.shape[0]):            # one sample at a time
        for j in range(W.shape[1]):        # one output node at a time
            s = 0.0
            for i in range(X.shape[1]):    # a_kj = sum_i x_ki w_ij + b_j
                s += X[k, i] * W[i, j]
            A[k, j] = s + B[0, j]
    return A

def forward_matmul(X, W, B):
    return X @ W + B                       # all samples, all nodes at once

t0 = time.perf_counter(); A1 = forward_loop(X, W, B);   t_loop = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(100):
    A2 = forward_matmul(X, W, B)
t_mat = (time.perf_counter() - t0) / 100

print("same result:", np.allclose(A1, A2))
print(f"Python for loop : {t_loop * 1000:10.2f} ms")
print(f"X @ W + B       : {t_mat * 1000:10.4f} ms")
print(f"speed-up        : about {t_loop / t_mat:,.0f} times")
