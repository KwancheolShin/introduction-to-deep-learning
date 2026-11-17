# %% 0. Setup
# Section 11: tokens.  Byte pair encoding (BPE) builds a vocabulary of word pieces by counting only:
#   start from single characters and repeatedly merge the most frequent pair of neighbouring tokens.
#   Part A: a toy corpus of four words, merge by merge
#   Part B: 1,000 merges learned from the Penn Treebank; how words are cut into tokens
import time
import collections
from text_utils import ptb_path

END = "_"                                        # marks the end of a word

# %% 1. BPE: learning the merges, and cutting a word into tokens
def merge_pair(toks, a, b):
    """replace every neighbouring pair (a, b) in the token list by the single token a+b"""
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
            out.append(a + b); i += 2
        else:
            out.append(toks[i]); i += 1
    return out

def learn_bpe(word_counts, n_merges, verbose=False):
    """word_counts: {word: how often it appears}.  Returns the merges, in the order they were learned."""
    words = {tuple(w) + (END,): c for w, c in word_counts.items()}       # every word as a list of characters
    merges = []
    for step in range(n_merges):
        pairs = collections.Counter()
        for toks, c in words.items():                                    # count neighbouring pairs
            for a, b in zip(toks[:-1], toks[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        (a, b), n = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))    # the most frequent pair
        merges.append((a, b))
        words = {tuple(merge_pair(list(toks), a, b)): c for toks, c in words.items()}
        if verbose:
            print(f"merge {step + 1}: '{a}' + '{b}' -> '{a + b}'  (appears {n} times)   "
                  + "   ".join(" ".join(toks) for toks in words))
    return merges

def tokenize(word, merges):
    """cut a word into tokens by applying the merges in the order they were learned"""
    toks = list(word) + [END]
    for a, b in merges:
        toks = merge_pair(toks, a, b)
    return toks

# %% 2. Part A: four words
toy = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
print("start:", "   ".join(" ".join(tuple(w) + (END,)) for w in toy))
toy_merges = learn_bpe(toy, 8, verbose=True)
for w in ["lowest", "newer"]:                    # words that were not in the corpus
    print(f"{w} -> {tokenize(w, toy_merges)}")

# %% 3. Part B: the Penn Treebank
with open(ptb_path("train"), encoding="utf-8") as f:       # downloaded to ../data on the first run
    counts = collections.Counter(f.read().split())
del counts["<unk>"]                              # a placeholder for rare words, not a word
chars = set("".join(counts)) | {END}
print(f"PTB: {sum(counts.values())} words, {len(counts)} different words, {len(chars)} different characters")

t0 = time.time()
n_merges = 1000
merges = learn_bpe(counts, n_merges)
print(f"{n_merges} merges learned in {time.time() - t0:.0f} s;  vocabulary = {len(chars)} characters + {len(merges)} merged tokens"
      f" = {len(chars) + len(merges)} tokens")
print("first 10 merges:", ["+".join(m) for m in merges[:10]])

cache = {w: tokenize(w, merges) for w in counts}
n_tok = sum(len(cache[w]) * c for w, c in counts.items())
n_chr = sum((len(w) + 1) * c for w, c in counts.items())
print(f"tokens per word in the corpus: {n_chr / sum(counts.values()):.2f} as characters, "
      f"{n_tok / sum(counts.values()):.2f} after {n_merges} merges")
print("words that became a single token:", sum(len(t) == 1 for t in cache.values()), "of", len(counts))

# %% 4. How are words cut?  Frequent words, rare words, and words that are not in the corpus at all
for w in ["the", "company", "million", "government", "international", "toyota", "galvanized", "weyerhaeuser"]:
    print(f"{w:14s} (appears {counts[w]:6d} times) -> {' '.join(cache[w])}")
for w in ["unbelievable", "tokenization", "chatgpt", "transformers"]:
    print(f"{w:14s} (not in the corpus: {w not in counts}) -> {' '.join(tokenize(w, merges))}")
