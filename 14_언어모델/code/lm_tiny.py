# %% 0. Setup
# Small examples that can be followed by hand; every number of these examples in the notes comes from this file.
#   1. self-attention built step by step on "the cat sat": average -> similarity -> query/key -> value
#   2. self-attention does not see the order of the words; adding position vectors fixes it
#   3. softmax with temperature
#   4. LayerNorm of one vector
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

# %% 1. Self-attention, step by step, on "the cat sat"
# word vectors with two features: (noun-like, verb-like)
X = np.array([[0., 0.],      # the
              [1., 0.],      # cat
              [0., 1.]])     # sat
causal = np.tril(np.ones((3, 3), dtype=bool))    # word i may look at the words j <= i

# step 1: plain average of the words up to i
A_avg = causal / causal.sum(axis=1, keepdims=True)
print("step 1, averaging weights:\n", A_avg)
print("step 1, outputs A X:\n", A_avg @ X)

# step 2: weights from similarity x_i . x_j (softmax over j <= i)
S_sim = np.where(causal, X @ X.T, -np.inf)
A_sim = softmax(S_sim)
print("step 2, similarity scores:\n", S_sim)
print("step 2, weights:\n", A_sim)

# step 3: separate query and key; a verb looks for a noun
W_Q = np.array([[0., 0.],    # noun feature -> looks for nothing in particular
                [2., 0.]])   # verb feature -> looks for the noun feature
W_K = np.eye(2)              # key = the word's own features
Q, K = X @ W_Q, X @ W_K
S_qk = np.where(causal, Q @ K.T, -np.inf)
A_qk = softmax(S_qk)
print("step 3, Q:\n", Q)
print("step 3, scores Q K^T:\n", S_qk)
print("step 3, weights:\n", A_qk)

# step 4: values (here W_V = identity: pass on the word's features)
W_V = np.eye(2)
O = A_qk @ (X @ W_V)
print("step 4, outputs A V:\n", O)

# %% 2. The order of the earlier words is invisible -- unless position vectors are added
def attention(X):
    S = np.where(causal, (X @ W_Q) @ (X @ W_K).T, -np.inf)
    return softmax(S) @ (X @ W_V)

the, cat, sat = X
print("output at 'sat':  'the cat sat'", attention(np.array([the, cat, sat]))[2],
      "  'cat the sat'", attention(np.array([cat, the, sat]))[2])
P = np.array([[0.5, 0.], [0., 0.5], [-0.5, 0.]])   # position vectors of positions 1, 2, 3
print("inputs with positions, 'the cat sat':\n", np.array([the, cat, sat]) + P)
print("inputs with positions, 'cat the sat':\n", np.array([cat, the, sat]) + P)
print("output at 'sat' with positions:  'the cat sat'", attention(np.array([the, cat, sat]) + P)[2],
      "  'cat the sat'", attention(np.array([cat, the, sat]) + P)[2])

# %% 3. Softmax with temperature
z = np.array([2.0, 1.0, 0.0])
for tau in [0.5, 1.0, 2.0]:
    print(f"temperature {tau}: p = {softmax(z / tau)}")

# %% 4. LayerNorm of one vector (without the learned scale and shift)
x = np.array([1.0, 2.0, 3.0, 6.0])
mu, var = x.mean(), x.var()
print(f"x = {x}, mean {mu}, variance {var}, LayerNorm(x) = {(x - mu) / np.sqrt(var)}")
