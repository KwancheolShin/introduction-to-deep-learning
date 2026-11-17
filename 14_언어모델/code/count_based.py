# %% 0. Setup
# Section 2: word vectors by counting -- co-occurrence matrix, cosine similarity, PPMI, SVD.
#   Part A: the 8-word sentence "you say goodbye and i say hello ."
#   Part B: the Penn Treebank corpus (about 930,000 words, 10,000 distinct words), from ../data/
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from text_utils import preprocess, load_ptb, co_occurrence, ppmi, cos_similarity, most_similar

np.set_printoptions(precision=3, suppress=True)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# %% 1. Corpus and word IDs
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
print("corpus    :", corpus)
print("id_to_word:", id_to_word)

# %% 2. Co-occurrence matrix (window 1)
V = len(word_to_id)
C = co_occurrence(corpus, V, window=1)
print(C.astype(int))

# %% 3. Cosine similarity of the count vectors
print("cos(you, i)       =", round(cos_similarity(C[word_to_id["you"]], C[word_to_id["i"]]), 4))
print("cos(you, goodbye) =", round(cos_similarity(C[word_to_id["you"]], C[word_to_id["goodbye"]]), 4))
print("most similar to 'you':", [(w, round(s, 4)) for w, s in most_similar("you", word_to_id, id_to_word, C)])

# %% 4. PPMI
M = ppmi(C)
print(M)
print("PMI check: C(say, goodbye) =", C[1, 2], ", C(say) =", C[1].sum(), ", C(goodbye) =", C[2].sum(), ", N =", C.sum())

# %% 5. SVD: 7-dimensional PPMI vectors -> 2 dimensions
U, S, Vt = np.linalg.svd(M)
print("singular values:", S)
W2 = U[:, :2] * S[:2]                            # 2-dimensional word vectors (rows): first 2 columns of U S
for w, i in word_to_id.items():
    print(f"{w:8s}", W2[i])
print("how much of M is kept by 2 of the 7 dimensions:", round(float((S[:2]**2).sum() / (S**2).sum()), 3))

fig, ax = plt.subplots(figsize=(5.6, 4.2))
groups = {}                                      # words with the same 2-dim vector share one label
for w, i in word_to_id.items():
    groups.setdefault((round(W2[i, 0], 3), round(W2[i, 1], 3)), []).append(w)
side = {"you": (0, 10, "center"), "goodbye": (0, -20, "center"), "hello": (0, 10, "center"),
        "say": (-10, -4, "right"), "and": (10, -4, "left"), ".": (10, -4, "left")}   # label offsets (points)
for (x0, y0), ws in groups.items():
    dx, dy, ha = side[ws[0]]
    ax.scatter(x0, y0, s=45, color="tab:blue")
    ax.annotate(", ".join(ws), (x0, y0), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=13)
ax.set_xlabel("dimension 1", fontsize=12); ax.set_ylabel("dimension 2", fontsize=12)
ax.set_title("the 7 words as 2-dimensional vectors", fontsize=12)
ax.margins(0.22)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/count_svd_2d.png", dpi=200)
plt.show()

# %% 6. Penn Treebank: co-occurrence (window 2) -> PPMI -> SVD (100 dimensions)
corpus, word_to_id, id_to_word = load_ptb("train")
V = len(word_to_id)
print(f"PTB: {len(corpus)} words, {V} distinct words")
t0 = time.time()
C = co_occurrence(corpus, V, window=2)
M = ppmi(C)
U, S, Vt = randomized_svd(M, n_components=100, n_iter=5, random_state=0)
W = U * S                          # 100-dimensional word vectors (rows)
print(f"co-occurrence, PPMI, SVD: {time.time() - t0:.0f} s")
for q in ["you", "year", "car", "toyota"]:
    print(q, "->", [(w, round(s, 3)) for w, s in most_similar(q, word_to_id, id_to_word, W)])
