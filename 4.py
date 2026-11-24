import nltk
from nltk.stem import PorterStemmer

# Create a Porter Stemmer object
stemmer = PorterStemmer()

# Sample words
words = ["running", "jumps", "studies", "studying", "easily", "fairly"]

print("Original Word -> Stemmed Word")
print("-" * 35)

# Apply stemming to each word
for word in words:
    stemmed_word = stemmer.stem(word)
    print(word, "->", stemmed_word)