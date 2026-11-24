import nltk
nltk.download('punkt_tab')  # Run once

from nltk.tokenize import word_tokenize

text = input("Enter a sentence: ")

# Character Tokenization
char_tokens = list(text)

# Word Tokenization
word_tokens = word_tokenize(text)

print("\nCharacter Tokens:")
print(char_tokens)

print("\nWord Tokens:")
print(word_tokens)