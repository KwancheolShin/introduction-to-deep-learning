# Helper functions shared by the scripts (the PTB corpus, co-occurrence matrix, PPMI, similarity).
import os
import urllib.request
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
PTB_URLS = ["https://raw.githubusercontent.com/wojzaremba/lstm/master/data/",     # two copies of the same files
            "https://raw.githubusercontent.com/tomsercu/lstm/master/data/"]


def ptb_path(split="train"):
    """Path of data/ptb.{split}.txt (Penn Treebank).  The file is downloaded the first time it is needed."""
    name = f"ptb.{split}.txt"
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        for url in PTB_URLS:
            try:
                print(f"downloading {name} from {url}")
                urllib.request.urlretrieve(url + name, path + ".part")
                os.replace(path + ".part", path)         # the file appears only when the download is complete
                break
            except OSError as e:
                print("  failed:", e)
        else:
            raise FileNotFoundError(f"could not download {name}; put the file in {os.path.abspath(DATA_DIR)}")
    return path


def preprocess(text):
    """'You say goodbye.' -> corpus [0, 1, 2, 3], word_to_id, id_to_word"""
    words = text.lower().replace(".", " .").split()
    word_to_id, id_to_word = {}, {}
    for w in words:
        if w not in word_to_id:
            word_to_id[w] = len(word_to_id)
            id_to_word[word_to_id[w]] = w
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def load_ptb(split="train"):
    """Penn Treebank: one sentence per line; each line end becomes the token <eos>."""
    with open(ptb_path(split), encoding="utf-8") as f:
        words = f.read().replace("\n", "<eos>").strip().split()
    word_to_id, id_to_word = {}, {}
    for w in words:
        if w not in word_to_id:
            word_to_id[w] = len(word_to_id)
            id_to_word[word_to_id[w]] = w
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def co_occurrence(corpus, vocab_size, window=1):
    """C[w, c] = how many times word c appears within `window` words of word w."""
    C = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    n = len(corpus)
    for k in range(1, window + 1):
        left, right = corpus[:-k], corpus[k:]
        np.add.at(C, (left, right), 1)      # right neighbour at distance k
        np.add.at(C, (right, left), 1)      # left neighbour at distance k
    return C


def ppmi(C, eps=1e-8):
    """PPMI(x, y) = max(0, log2( C(x,y) N / (C(x) C(y)) ))"""
    N = C.sum()
    S = C.sum(axis=0)
    M = np.log2(C * N / (S[:, None] * S[None, :]) + eps)
    return np.maximum(M, 0).astype(np.float32)


def cos_similarity(x, y, eps=1e-8):
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + eps))


def most_similar(query, word_to_id, id_to_word, W, top=5):
    """Words whose vectors (rows of W) have the largest cosine similarity with the query word."""
    q = W[word_to_id[query]]
    sims = W @ q / (np.linalg.norm(W, axis=1) * np.linalg.norm(q) + 1e-8)
    out = []
    for i in np.argsort(-sims):
        if id_to_word[i] == query:
            continue
        out.append((id_to_word[i], float(sims[i])))
        if len(out) == top:
            break
    return out
