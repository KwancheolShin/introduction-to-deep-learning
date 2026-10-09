# optimizers.py -- parameter update rules (Week: Techniques for Training)
# Every class has  update(params, grads):
#   params, grads : dictionaries with the same keys; params are updated IN PLACE
#   (the same form as network.params / grads in mnist_lab.py and mnist_mlp.py)

import numpy as np


class SGD:
    """W <- W - lr * dL/dW"""
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    """v <- beta * v - lr * dL/dW,   W <- W + v"""
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None

    def update(self, params, grads):
        if self.v is None:                          # velocity starts at 0
            self.v = {key: np.zeros_like(val) for key, val in params.items()}
        for key in params:
            self.v[key] = self.beta * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """h <- h + g*g,   W <- W - lr * g / (sqrt(h) + eps)"""
    def __init__(self, lr=0.01, eps=1e-7):
        self.lr = lr
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}
        for key in params:
            self.h[key] += grads[key] * grads[key]           # sum of squared gradients
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)


class RMSProp:
    """h <- rho * h + (1 - rho) * g*g,   W <- W - lr * g / (sqrt(h) + eps)"""
    def __init__(self, lr=0.01, rho=0.9, eps=1e-7):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}
        for key in params:
            self.h[key] = self.rho * self.h[key] + (1 - self.rho) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.eps)


class Adam:
    """m <- b1*m + (1-b1)*g,   v <- b2*v + (1-b2)*g*g,
       W <- W - lr * m_hat / (sqrt(v_hat) + eps)   (m_hat, v_hat: bias-corrected)"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}
        self.k += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.k)     # bias correction
            v_hat = self.v[key] / (1 - self.beta2 ** self.k)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam + decoupled weight decay:  W <- W - lr * (m_hat / (sqrt(v_hat) + eps) + wd * W)
       The decay does NOT go through m and v.  Only weights (keys "W1", "W2", ...) are decayed."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    def update(self, params, grads):
        for key in params:
            if key.startswith("W"):
                params[key] -= self.lr * self.weight_decay * params[key]   # shrink W
        super().update(params, grads)                    # the usual Adam step
