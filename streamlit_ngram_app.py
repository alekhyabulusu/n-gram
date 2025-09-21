import streamlit as st
import nltk
from nltk.corpus import gutenberg
nltk.download('gutenberg')
nltk.download('punkt_tab')
nltk.download('punkt')
import re
from collections import Counter
import math

st.set_page_config(page_title="N-gram Autocomplete", layout="centered")

@st.cache_data(show_spinner=False)
def download_and_prepare():
    nltk.download('gutenberg', quiet=True)
    nltk.download('punkt', quiet=True)
    raw_text = gutenberg.raw('austen-sense.txt')
    text = raw_text.lower()
    text = re.sub(r"[^a-z\s']", ' ', text)
    tokens = nltk.word_tokenize(text)
    return tokens

@st.cache_data(show_spinner=False)
def build_ngrams(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_freq = Counter(ngrams)
    context_freq = Counter([ng[:-1] for ng in ngrams])
    return ngram_freq, context_freq

def word_prob(word, context, ngram_freq, context_freq, vocab_size):
    context = tuple(context)
    ngram = context + (word,)
    numerator = ngram_freq.get(ngram, 0) + 1  
    denominator = context_freq.get(context, 0) + vocab_size
    return numerator / denominator

@st.cache_data(show_spinner=False)
def get_vocab(tokens):
    return sorted(list(set(tokens)))

def predict_next_word(context, ngram_freq, context_freq, vocab, top_k=1):
    probs = []
    vocab_size = len(vocab)
    for w in vocab:
        p = word_prob(w, context, ngram_freq, context_freq, vocab_size)
        probs.append((w, p))
    probs.sort(key=lambda x: x[1], reverse=True)
    return probs[:top_k]

def generate_sentence(prefix_tokens, length, n, ngram_freq, context_freq, vocab):
    tokens = list(prefix_tokens)
    for _ in range(length):
        context = tokens[-(n-1):] if len(tokens) >= (n-1) else tokens
        top = predict_next_word(context, ngram_freq, context_freq, vocab, top_k=1)
        next_word = top[0][0]
        tokens.append(next_word)
    return ' '.join(tokens)

# --- App layout ---
st.title("N-gram Autocomplete")

with st.spinner('Preparing corpus...'):
    tokens = download_and_prepare()

col1, col2 = st.columns([2,1])
with col1:
    prefix = st.text_input("Prefix (words)", value="")
    length = st.number_input("Number of words to generate", min_value=1, max_value=50, value=10)
    n = st.selectbox("Choose n (n-gram order)", options=[2,3,4,5], index=1)

#with col2:
    #st.write("Model building settings")
    #use_full = st.checkbox("Train on full corpus (recommended for UI)", value=True)
    #split_ratio = st.slider("If not full corpus: train ratio", 0.1, 0.95, 0.8)

# Build model
train_tokens = tokens
split = int(len(tokens) * 0.7)
train_tokens = tokens[:split]

#if not use_full:
 #   split = int(len(tokens) * float(split_ratio))
  #  train_tokens = tokens[:split]

with st.spinner('Building n-gram counts...'):
    ngram_freq, context_freq = build_ngrams(train_tokens, n)
    vocab = get_vocab(train_tokens)

st.markdown("---")

if st.button("Generate completion"):
    # tokenize prefix (simple whitespace tokenize)
    prefix_tokens = re.findall(r"[a-z']+", prefix.lower())
    if any(not t for t in prefix_tokens):
        st.error("Prefix must contain words (letters/apostrophes).")
    else:
        # Generate
        generated = generate_sentence(prefix_tokens, length, n, ngram_freq, context_freq, vocab)
        st.subheader("Generated completion")
        st.write(generated)
        st.markdown("---")
        

