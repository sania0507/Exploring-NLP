import nltk
from nltk.stem import WordNetLemmatizer

# Download required dataset
nltk.download('wordnet')

# Create lemmatizer object
lemmatizer = WordNetLemmatizer()

# Sample text
text = "The boys are running and the girls are studying better."

# Tokenize the text
words = text.split()

print("Original Words:")
print(words)

print("\nLemmatized Words:")

for word in words:
    lemma = lemmatizer.lemmatize(word)
    print(word, "->", lemma)