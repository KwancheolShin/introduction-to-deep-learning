# mnist_data.py -- load MNIST, flatten, normalize, one-hot, train/validation split

import numpy as np


def load_mnist():
    """Return (x_train, t_train), (x_test, t_test): images (N,28,28), labels (N,)."""
    try:
        from tensorflow.keras.datasets import mnist   # if TensorFlow is installed
        return mnist.load_data()
    except ImportError:
        from sklearn.datasets import fetch_openml     # otherwise download from OpenML
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        X = X.reshape(-1, 28, 28).astype(np.uint8)
        y = y.astype(np.int64)
        return (X[:60000], y[:60000]), (X[60000:], y[60000:])


def load_mnist_split(n_val=5000, seed=0):
    """Return x_train, t_train, x_val, t_val, x_test, t_test  (images flattened to 784, labels one-hot).
    The same split as mnist_mlp.py when seed=0."""
    np.random.seed(seed)
    (x_all_o, t_all_o), (x_test_o, t_test_o) = load_mnist()

    x_all  = x_all_o.reshape(x_all_o.shape[0], -1) / 255.0      # (60000, 784)
    x_test = x_test_o.reshape(x_test_o.shape[0], -1) / 255.0    # (10000, 784)
    t_all  = np.eye(10)[t_all_o]                                 # (60000, 10)
    t_test = np.eye(10)[t_test_o]                                # (10000, 10)

    # hold out n_val training images as a VALIDATION set
    perm = np.random.permutation(x_all.shape[0])
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (x_all[train_idx], t_all[train_idx],
            x_all[val_idx], t_all[val_idx],
            x_test, t_test)
