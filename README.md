# Exploring NLP

A collection of hands-on Natural Language Processing exercises covering the classic
pipeline — from raw text cleaning and tokenization through to modern transformer
embeddings and speech-to-text. Each script is self-contained, heavily commented,
and runnable on its own.

## Contents

| File | Topic | Libraries |
| --- | --- | --- |
| `1.py` | Character & word tokenization | `nltk` |
| `2.py` | Hindi speech-to-text, then translation to English | `openai-whisper`, `deep-translator` |
| `3.py` | Lemmatization | `nltk` (WordNet) |
| `4.py` | Stemming (Porter stemmer) | `nltk` |
| `5.py` | Bag of Words & TF-IDF vectorization | `scikit-learn` |
| `6.py` | Static word embeddings: Word2Vec and GloVe | `gensim`, `numpy` |
| `7.py` | Contextual embeddings from BERT + mean pooling | `transformers`, `torch` |
| `8.py` | Regex-based text preprocessing pipeline | `re` (stdlib) |
| `9.py` | Part-of-speech tagging with a supervised HMM | `nltk` |

### What each script demonstrates

**`1.py` — Tokenization.** Splits a user-supplied sentence into character tokens
(plain `list(text)`) and word tokens (`nltk.word_tokenize`), showing how the two
granularities differ.

**`2.py` — Speech-to-text + machine translation.** Loads OpenAI's Whisper `base`
ASR model, transcribes `hindi_audio2.m4a` with `language="hi"`, then passes the
Hindi transcript through `GoogleTranslator` to produce English. Requires network
access for the translation step.

**`3.py` — Lemmatization.** Uses `WordNetLemmatizer` on a sample sentence. Because
no POS tag is passed, words are assumed to be nouns, so `"running"` stays
`"running"` while `"boys"` becomes `"boy"`.

**`4.py` — Stemming.** Applies `PorterStemmer` to a word list, comparing crude
suffix-stripping (`"studies"` → `"studi"`, `"easily"` → `"easili"`) against
lemmatization's dictionary-based approach.

**`5.py` — Vectorization.** Builds a Bag-of-Words count matrix and a TF-IDF matrix
over three sample documents with scikit-learn, printing both vocabularies and the
resulting dense matrices.

**`6.py` — Static embeddings.** Trains a small `Word2Vec` model on four sentences
and queries nearest neighbours, then loads the pre-trained GloVe vectors from
`glove.6B.100d.txt` for comparison. See [Datasets](#datasets) — this file is not
included in the repo.

**`7.py` — Contextual embeddings.** Feeds three sentences through
`bert-base-uncased`, then mean-pools the token embeddings (masked by the attention
mask so padding is excluded) to get one 768-dimensional vector per sentence. The
denominator is clamped to avoid dividing by zero on empty rows.

**`8.py` — Text preprocessing.** A regex pipeline applied in order: lowercase, strip
URLs, emails, `@mentions` and `#hashtags`, punctuation, digits, and collapse
whitespace — then tokenize on whitespace.

**`9.py` — POS tagging with an HMM.** Trains a `HiddenMarkovModelTrainer` on 3,000
Treebank sentences using a `LidstoneProbDist` estimator, tags a held-out slice, and
reports accuracy.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install nltk scikit-learn gensim numpy transformers torch openai-whisper deep-translator
```

Whisper additionally needs **FFmpeg** on your `PATH` to decode `.m4a` audio:

```bash
# Windows
winget install Gyan.FFmpeg
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
```

## Running

Each script is standalone:

```bash
python 1.py
```

Scripts `1`, `3`, and `9` call `nltk.download(...)` on first run to fetch the
`punkt_tab`, `wordnet`, `treebank`, and `universal_tagset` resources. Subsequent
runs reuse the cached data.

`2.py` expects `hindi_audio2.m4a` in the working directory — it is included in this
repo. `6.py` expects `glove.6B.100d.txt` (see below).

## Datasets

**`hindi_audio2.m4a`** is committed, so `2.py` runs out of the box.

**`glove.6B.100d.txt`** is **not** committed — it is 347 MB, well over GitHub's
100 MB per-file limit. Download it from the
[Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/) and place it
next to `6.py`:

```bash
# From the glove.6B.zip archive
unzip glove.6B.zip glove.6B.100d.txt
```

Without this file `6.py` raises `FileNotFoundError` at the GloVe section (the
Word2Vec portion above it still runs).

To track the file with Git instead, use
[Git LFS](https://git-lfs.github.com/): `git lfs track "glove.6B.100d.txt"` and
remove the line from `.gitignore`.

## Notes

- `.gitignore` excludes `glove.6B.100d.txt`, virtual environments, NLTK downloads,
  and `tempCodeRunnerFile.py` (a VS Code scratch file).
- `7.py` downloads the ~440 MB `bert-base-uncased` weights on first run, cached
  under `~/.cache/huggingface`.
- `9.py`'s HMM is trained on a subset for speed; accuracy improves with more
  training data, so treat the printed figure as illustrative rather than
  state-of-the-art.
