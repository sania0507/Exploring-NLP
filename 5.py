# Bag of words ,TF IDF

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample documents
documents = [
    "I love machine learning",
    "I love natural language processing",
    "Machine learning is interesting"
]

# ---------------- BAG OF WORDS ----------------

bow = CountVectorizer()

bow_matrix = bow.fit_transform(documents)

print("Bag of Words Vocabulary:")
print(bow.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())


# ---------------- TF-IDF ----------------

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(documents)

print("\nTF-IDF Vocabulary:")
print(tfidf.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())