# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import scipy as sp
import random
import numpy as np
import re
from datasets import load_dataset
import matplotlib.pyplot as plt
import pprint
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from platform import python_version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 8

assert int(python_version().split(".")[1]) >= 5, "Please upgrade your Python version following the instructions in \
    the README.md file found in the same directory as this notebook. Your Python version is " + python_version()

plt.rcParams['figure.figsize'] = [10, 5]

# Only load dataset if running as main script
if __name__ == "__main__":
    imdb_dataset = load_dataset("stanfordnlp/imdb", name="plain_text")
else:
    imdb_dataset = None


START_TOKEN = '<START>'
END_TOKEN = '<END>'
NUM_SAMPLES = 150

np.random.seed(0)
random.seed(0)
# ----------------


# ---------------------------------------------
# PART 1: Count-Based Word Vectors
# ---------------------------------------------

# pre-process: extract words, lowercase and add begin & end token
def read_corpus():
    """Read files from the Large Movie Review Dataset.
    Params:
        category (string): category name
    Returns:
        list of lists, with words from each of the processed files
    """
    global imdb_dataset
    if imdb_dataset is None:
        imdb_dataset = load_dataset("stanfordnlp/imdb", name="plain_text")
    files = imdb_dataset["train"]["text"][:NUM_SAMPLES]
    return [[START_TOKEN] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [END_TOKEN] for f in files]


# imdb_corpus = read_corpus()
# pprint.pprint(imdb_corpus[:3], compact=True, width=100)
# print("corpus size: ", len(imdb_corpus[0]))


# work out the distinct words in the corpus
def distinct_words(corpus: list) -> list:
    """Determine a list of distinct words for the corpus.
    Params:
        corpus (list of list of strings): corpus of documents
    Returns:
        corpus_words (list of strings): sorted list of distinct words across the corpus
        n_corpus_words (integer): number of distinct words across the corpus
    """
    flattened_corpus = [y for x in corpus for y in x]
    word_set = {x for x in flattened_corpus}

    corpus_words = sorted(word_set)
    n_corpus_words = len(corpus_words)

    return corpus_words, n_corpus_words


def compute_co_occurrence_matrix(corpus: list, window_size: int = 4):
    """Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    Note:
        Each word in a document should be at the center of a window. Words near edges will have
        a smaller number of co-occurring words.

    Params:
        corpus (list of list of strings): corpus of documents
        window_size (int): size of context window
    Returns:
        M (a symmetric numpy matrix of shape (n_corpus_words, n_corpus_words)):
            Co-occurence matrix of word counts.
            The ordering of the words in the rows/cols should be the same as the ordering
            of the words given by distinct_words function.
        word2ind (dict): dictionary that maps word to index for matrix M.
    """
    words, n_words = distinct_words(corpus)
    M = np.zeros((n_words, n_words), dtype=np.int32)
    word2ind = {word: i for i, word in enumerate(words)}

    for f in corpus:
        f_len = len(f)
        for i in range(f_len):
            center_idx = word2ind[f[i]]
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, f_len)

            for j in range(start, i):
                context_idx = word2ind[f[j]]
                M[center_idx, context_idx] += 1
            for j in range(i + 1, end):
                context_idx = word2ind[f[j]]
                M[center_idx, context_idx] += 1
    return M, word2ind


def reduce_to_k_dim(M: np.ndarray, k: int = 2):
    """
    Reduce a co-occurence count matrix of dimensionality (n_corpus_words, n_corpus_words)
    to a matrix of dimensionality (n_corpus_words, k) using SVD function from Scikit-Learn

    Params:
        M (np.ndarray): co-occurence matrix of shape (n_corpus_words, n_corpus_words)
        k (int): embedding size of each word after dimentsion reduction

    Returns:
        M_reduced (np.ndarray): matrix of k-dimensional word embeddings of shape (n_corpus_words, k)
    """
    n_iters = 10
    print(f"Running Truncated SVD over {M.shape[0]} words ...")

    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=n_iters)
    M_reduced = svd.fit_transform(M)

    print("Done.")
    # print("Original:")
    # print(M)
    # print("Transformed:")
    # print(M_reduced)
    return M_reduced


def plot_embeddings(M_reduced: np.ndarray, word2ind: dict, words: list) -> None:
    """
    Plot in a scatterplot the embeddings of the words specified in the list "words".
    NOTE: do not plot all the words listed in M_reduced / word2ind.
    Include a label next to each point.

    Params:
        M_reduced (np.ndarray): matrix of 2-dimensional word embeddings
        word2ind (dict): dictionary that maps word to indices for matrix M
        words (list of strings): words whose embeddings we want to visualize
    """

    for word in words:
        if word in word2ind:
            idx = word2ind[word]
            x, y = M_reduced[idx]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)

    plt.show()


# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------
# imdb_corpus = read_corpus()
# M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(
#     imdb_corpus)
# M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
#
# # Rescale (normalize) the rows to make them each of unit-length
# M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
# M_normalized = M_reduced_co_occurrence / \
#     M_lengths[:, np.newaxis]  # broadcasting
#
# words = ['movie', 'book', 'mysterious', 'story', 'fascinating',
#          'good', 'interesting', 'large', 'massive', 'huge']
#
# plot_embeddings(M_normalized, word2ind_co_occurrence, words)


# ---------------------------------------------
# PART 2: Prediction-Based Word Vectors
# ---------------------------------------------

def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each length 200
    """
    import gensim.downloader as api
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


wv_from_bin = load_embedding_model()


def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the GloVe vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 400000 GloVe vectors loaded from file
        Return:
            M: numpy matrix shape (num words, 200) containing the vectors
            word2ind: dictionary mapping each word to its row number in M
    """
    import random
    words = list(wv_from_bin.index_to_key)
    print("Shuffling words ...")
    random.seed(225)
    random.shuffle(words)
    print(f"Putting {len(words)} words into word2ind and matrix M...")
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        if w in words:
            continue
        try:
            M.append(wv_from_bin.get_vector(w))
            word2ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2ind


# -----------------------------------------------------------------
# Run Cell to Reduce 200-Dimensional Word Embeddings to k Dimensions
# Note: This should be quick to run
# -----------------------------------------------------------------
words = ['movie', 'book', 'mysterious', 'story', 'fascinating',
         'good', 'interesting', 'large', 'massive', 'huge']
M, word2ind = get_matrix_of_vectors(wv_from_bin, words)
M_reduced = reduce_to_k_dim(M, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis]  # broadcasting

plot_embeddings(M_reduced_normalized, word2ind, words)

# Question 2.2
similar_words = wv_from_bin.most_similar('shit', topn=10)
print(similar_words)

# Question 2.3
print(wv_from_bin.most_similar('leave', topn=10))
print(
    f"distance between 'leave' and 'stay': {wv_from_bin.distance('leave', 'stay')}")
print(
    f"distance between 'leave' and 'go': {wv_from_bin.distance('leave', 'go')}")

# Question 2.4
# Run this cell to answer the analogy -- man : grandfather :: woman : x
pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'grandfather'], negative=['man']))

# Question 2.5
x, y, a, b = ("male", "female", "man", "woman")
