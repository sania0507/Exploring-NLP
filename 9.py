import nltk
from nltk.corpus import treebank
from nltk.tag import hmm
from nltk.probability import LidstoneProbDist

# Download dataset
nltk.download('treebank')
nltk.download('universal_tagset')

# Load tagged sentences
sentences = treebank.tagged_sents(tagset='universal')

# Split dataset into training and testing
train_data = sentences[:3000]
test_data = sentences[3000:3100]

# Train Hidden Markov Model
trainer = hmm.HiddenMarkovModelTrainer()

estimator = lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)

model = trainer.train_supervised(
    train_data,
    estimator=estimator
)

# Test the model
for sentence in test_data[:5]:
    words = [word for word, tag in sentence]

    # model is a TaggerI: tagging is done via .tag(), not .pos_tag()
    predicted = model.tag(words)

    print("\nSentence:", words)
    print("Predicted Tags:", predicted)
    print("Actual Tags:", sentence)

# Evaluate the model
accuracy = model.accuracy(test_data)

print("\nModel Accuracy:", accuracy)
