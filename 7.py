from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Input sentences
sentences = [
    "Text classification is a core task in NLP.",
    "Word embeddings turn words into dense numeric vectors.",
    "Deep learning models improve as more data becomes available."
]

# Tokenize the sentences
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Generate BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling to generate sentence vectors
token_embeddings = outputs.last_hidden_state
attention_mask = inputs["attention_mask"].unsqueeze(-1)

masked_embeddings = token_embeddings * attention_mask

# Clamp the denominator so a fully-padded/empty row cannot divide by zero
token_counts = attention_mask.sum(dim=1).clamp(min=1e-9)
sentence_embeddings = masked_embeddings.sum(dim=1) / token_counts

# Display vectorized output
print("Sentence Embeddings:")
print(sentence_embeddings)

print("\nEmbedding Shape:", sentence_embeddings.shape)
