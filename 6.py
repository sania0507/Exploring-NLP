# Static Word Embedding using Word2Vec and GloVe

from gensim.models import Word2Vec
import numpy as np

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is interesting",
    "Natural language processing is a part of machine learning",
    "I love natural language processing"
]

# Tokenize the documents
sentences = [sentence.lower().split() for sentence in documents]

# ---------------- WORD2VEC ----------------

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

print("Word2Vec vector for 'machine':")
print(model.wv["machine"])

print("\nSimilar words to 'machine':")
print(model.wv.most_similar("machine"))


# ---------------- GLOVE ----------------

# GloVe pretrained embeddings
# Download glove.6B.100d.txt from:
# https://nlp.stanford.edu/projects/glove/

glove_file = "glove.6B.100d.txt"

glove_vectors = {}

with open(glove_file, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove_vectors[word] = vector

print("\nGloVe vector for 'machine':")
print(glove_vectors["machine"])