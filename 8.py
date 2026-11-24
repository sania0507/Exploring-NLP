import re

original_text = (
    "Good morning!!! This is a demo of NLP preprocessing. "
    "Read https://sample.org/blog for the full write-up. "
    "Contact: student99@sample.org #NLP2024 @lecturer"
)

text = original_text

# 1. Convert text to lowercase
text = text.lower()

# 2. Remove URLs
text = re.sub(r'http\S+|www\S+', '', text)

# 3. Remove email addresses
text = re.sub(r'\S+@\S+', '', text)

# 4. Remove mentions and hashtags
text = re.sub(r'@\w+|#\w+', '', text)

# 5. Remove punctuation and special characters
text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

# 6. Remove numbers
text = re.sub(r'\d+', '', text)

# 7. Remove extra whitespaces
text = re.sub(r'\s+', ' ', text).strip()

# 8. Tokenization
tokens = text.split()

print("Original Text:", original_text)
print("Preprocessed Text:", text)
print("Tokens:", tokens)
