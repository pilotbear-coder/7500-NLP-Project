"""
birnn.py
========
Tier 2 — Bidirectional LSTM with Pre-trained Word Embeddings
Dataset: SST-2 (binary) and SST-5 (fine-grained)

Concepts demonstrated:
  - Tokenization (spaCy)
  - Word2Vec and GloVe embeddings (pre-trained, frozen vs fine-tuned)
  - Recurrent Neural Networks (LSTM)
  - Bidirectional RNN (BiLSTM)
  - First-order optimisation (Adam) + gradient clipping
  - Gradient flow analysis (vanishing gradient diagnostic)
  - Evaluation: accuracy, macro-F1, confusion matrix, learning curves

Usage:
    from models.birnn import BiLSTMClassifier, Vocabulary, run_birnn_experiment
    results = run_birnn_experiment(version=2, embedding='glove')
"""

import os
import re
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# ── Confirmed dataset field names ────────────────────────────────────────────
# stanfordnlp/sst2 → 'sentence' | SetFit/sst5 → 'text'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')


# ════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING  (same function as baseline.py and 01_eda.ipynb)
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean SST text for N-gram / BiRNN tiers (NOT for BERT).
    Fixes PTB bracket tokens, split contractions, and whitespace.
    """
    text = text.strip()
    text = re.sub(r'-lrb-', '(', text, flags=re.IGNORECASE)
    text = re.sub(r'-rrb-', ')', text, flags=re.IGNORECASE)
    text = re.sub(r" n 't", "n't", text)
    text = re.sub(r" 's",   "'s",  text)
    text = re.sub(r" 're",  "'re", text)
    text = re.sub(r" 've",  "'ve", text)
    text = re.sub(r" 'll",  "'ll", text)
    text = re.sub(r" 'd",   "'d",  text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_sst(version: int = 2):
    """
    Load SST-2 or SST-5 from HuggingFace and apply cleaning.

    Returns: train_texts, train_labels, val_texts, val_labels, label_names
    """
    assert version in (2, 5), "version must be 2 or 5"

    if version == 2:
        ds = load_dataset('stanfordnlp/sst2')
        train_texts  = [clean_text(t) for t in ds['train']['sentence']]
        train_labels = list(ds['train']['label'])
        val_texts    = [clean_text(t) for t in ds['validation']['sentence']]
        val_labels   = list(ds['validation']['label'])
        label_names  = ['negative', 'positive']
    else:
        ds = load_dataset('SetFit/sst5')
        train_texts  = [clean_text(t) for t in ds['train']['text']]
        train_labels = list(ds['train']['label'])
        val_texts    = [clean_text(t) for t in ds['validation']['text']]
        val_labels   = list(ds['validation']['label'])
        label_names  = ['very negative', 'negative', 'neutral',
                        'positive', 'very positive']

    print(f'\n── SST-{version} loaded ──────────────────────────────────')
    print(f'  Train : {len(train_texts):,} | Val: {len(val_texts):,}')
    print(f'  Labels: {label_names}')
    return train_texts, train_labels, val_texts, val_labels, label_names


# ════════════════════════════════════════════════════════════════════════════
# 3. TOKENISATION (spaCy)
# ════════════════════════════════════════════════════════════════════════════

def spacy_tokenize(texts: list, batch_size: int = 512) -> list:
    """
    Tokenise a list of strings using spaCy.

    Concept: spaCy's tokeniser handles punctuation, contractions, and
    hyphenated words more accurately than regex whitespace splitting —
    an explicit upgrade over the Tier 1 tokeniser.
    """
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'spacy', 'download',
                        'en_core_web_sm'], check=True)
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])

    tokenized = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokenized.append([token.text.lower() for token in doc
                          if not token.is_space])
    return tokenized


# ════════════════════════════════════════════════════════════════════════════
# 4. VOCABULARY
# ════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Maps tokens to integer indices. Handles special tokens:
      <PAD> = 0  (padding to fixed length)
      <UNK> = 1  (out-of-vocabulary tokens)
    """
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2token = {0: '<PAD>', 1: '<UNK>'}

    def build(self, tokenized_texts: list):
        """Build vocabulary from a list of token lists."""
        counter = Counter(tok for tokens in tokenized_texts for tok in tokens)
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        print(f'  Vocabulary: {len(self.token2idx):,} tokens '
              f'(min_freq={self.min_freq})')
        return self

    def encode(self, tokens: list) -> list:
        return [self.token2idx.get(t, self.UNK_IDX) for t in tokens]

    def __len__(self):
        return len(self.token2idx)


# ════════════════════════════════════════════════════════════════════════════
# 5. EMBEDDING LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_glove(glove_path: str, vocab: Vocabulary,
               embedding_dim: int = 100) -> torch.Tensor:
    """
    Load GloVe embeddings (glove.6B.100d or glove.6B.300d) and build
    an embedding matrix aligned to our vocabulary.

    Concept: GloVe (Global Vectors) embeddings are trained by factorising
    a global word co-occurrence matrix. Unlike Word2Vec (local context
    windows), GloVe incorporates corpus-wide statistics directly.

    Download: https://nlp.stanford.edu/data/glove.6B.zip

    OOV tokens: initialised with small random vectors (mean ≈ 0, std ≈ 0.01)
    so they are near the origin and distinguishable from real embeddings.
    """
    print(f'  Loading GloVe from {glove_path}...')
    glove_vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            if len(vec) == embedding_dim:
                glove_vectors[word] = vec

    embedding_matrix = np.random.normal(
        scale=0.01, size=(len(vocab), embedding_dim)
    ).astype(np.float32)
    embedding_matrix[vocab.PAD_IDX] = 0.0   # <PAD> stays zero

    n_found = 0
    for token, idx in vocab.token2idx.items():
        if token in glove_vectors:
            embedding_matrix[idx] = glove_vectors[token]
            n_found += 1

    coverage = n_found / max(len(vocab) - 2, 1) * 100
    print(f'  GloVe coverage: {n_found:,}/{len(vocab)-2:,} tokens ({coverage:.1f}%)')
    return torch.tensor(embedding_matrix)


def load_word2vec(w2v_path: str, vocab: Vocabulary,
                  embedding_dim: int = 300) -> torch.Tensor:
    """
    Load Word2Vec embeddings (Google News 300d binary or gensim KeyedVectors).

    Concept: Word2Vec (Skip-gram / CBOW) learns embeddings by predicting
    surrounding words from a local context window. It captures syntactic
    and semantic regularities (e.g. king - man + woman ≈ queen).

    Download (Google News): https://code.google.com/archive/p/word2vec/
    Or via gensim: gensim.downloader.load('word2vec-google-news-300')

    OOV tokens: same random init strategy as GloVe.
    """
    try:
        from gensim.models import KeyedVectors
        print(f'  Loading Word2Vec from {w2v_path}...')
        is_binary = w2v_path.endswith('.bin')
        w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=is_binary)
    except ImportError:
        raise ImportError('gensim is required for Word2Vec: pip install gensim')

    embedding_matrix = np.random.normal(
        scale=0.01, size=(len(vocab), embedding_dim)
    ).astype(np.float32)
    embedding_matrix[vocab.PAD_IDX] = 0.0

    n_found = 0
    for token, idx in vocab.token2idx.items():
        if token in w2v:
            embedding_matrix[idx] = w2v[token]
            n_found += 1

    coverage = n_found / max(len(vocab) - 2, 1) * 100
    print(f'  Word2Vec coverage: {n_found:,}/{len(vocab)-2:,} tokens ({coverage:.1f}%)')
    return torch.tensor(embedding_matrix)


def make_random_embeddings(vocab: Vocabulary,
                           embedding_dim: int = 100) -> torch.Tensor:
    """Fallback: random embeddings when no pre-trained file is available."""
    matrix = torch.randn(len(vocab), embedding_dim) * 0.01
    matrix[vocab.PAD_IDX] = 0.0
    print(f'  Using random embeddings ({embedding_dim}d) — '
          f'for ablation / no pre-trained file available.')
    return matrix


# ════════════════════════════════════════════════════════════════════════════
# 6. PYTORCH DATASET
# ════════════════════════════════════════════════════════════════════════════

class SSTDataset(Dataset):
    """
    PyTorch Dataset for SST-2 / SST-5.
    Encodes token lists to integer sequences and pads/truncates to max_len.
    """

    def __init__(self, tokenized_texts: list, labels: list,
                 vocab: Vocabulary, max_len: int = 64):
        self.max_len = max_len
        self.labels  = torch.tensor(labels, dtype=torch.long)

        # Encode and store lengths (for pack_padded_sequence)
        self.encoded = []
        self.lengths = []
        for tokens in tokenized_texts:
            ids = vocab.encode(tokens)[:max_len]
            self.lengths.append(len(ids))
            # Pad to max_len
            ids += [vocab.PAD_IDX] * (max_len - len(ids))
            self.encoded.append(ids)

        self.encoded = torch.tensor(self.encoded, dtype=torch.long)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded[idx], self.lengths[idx], self.labels[idx]


# ════════════════════════════════════════════════════════════════════════════
# 7. BiLSTM MODEL
# ════════════════════════════════════════════════════════════════════════════

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sentence classification.

    Architecture:
      Embedding → Dropout → BiLSTM (2 layers) → Max-pool over time → Dropout
      → Linear → (optional) Linear → Softmax

    Concept — Bidirectional RNN:
      A standard LSTM processes tokens left-to-right, so the hidden state at
      step t encodes only the left context. A BiLSTM runs two LSTMs — one
      forward (left→right) and one backward (right→left) — and concatenates
      their outputs. The final representation captures both past and future
      context for every token, which is critical for sentiment (e.g. negation
      at the end of a phrase affects the beginning).

    Concept — Max-over-time pooling:
      Instead of using only the last hidden state (which suffers from
      vanishing gradients in long sequences), we take the element-wise
      maximum across all time steps. This retains the most salient feature
      from any position in the sequence.
    """

    def __init__(self,
                 vocab_size:     int,
                 embedding_dim:  int,
                 hidden_dim:     int,
                 n_layers:       int,
                 n_classes:      int,
                 dropout:        float = 0.3,
                 pretrained_emb: torch.Tensor = None,
                 freeze_emb:     bool = False):
        super().__init__()

        # ── Embedding layer ─────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight = nn.Parameter(pretrained_emb)
            if freeze_emb:
                self.embedding.weight.requires_grad = False

        # ── BiLSTM ──────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = embedding_dim,
            hidden_size   = hidden_dim,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,            # forward + backward
            dropout       = dropout if n_layers > 1 else 0.0,
        )

        # ── Classification head ─────────────────────────────────────────────
        # BiLSTM output dim = hidden_dim * 2 (forward + backward concatenated)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim * 2, n_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for LSTM weights; zero bias."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 to encourage memory retention
                n = param.size(0)
                param.data[n//4 : n//2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (batch, max_len) token indices
        lengths : (batch,) actual sequence lengths (before padding)
        returns : (batch, n_classes) logits
        """
        # Embed
        embedded = self.dropout(self.embedding(x))  # (B, L, E)

        # Pack for efficient LSTM (skips PAD positions)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        output, _   = pad_packed_sequence(lstm_out, batch_first=True)
        # output: (B, L, hidden_dim*2)

        # Max-over-time pooling — keeps strongest signal from any position
        # Mask padding positions to -inf so they don't win the max
        mask = (x == 0).unsqueeze(-1)                   # (B, L, 1)
        output = output.masked_fill(mask, float('-inf'))
        pooled = output.max(dim=1).values               # (B, hidden_dim*2)

        # Classify
        logits = self.fc(self.dropout(pooled))           # (B, n_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════
# 8. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion,
                max_grad_norm: float = 5.0):
    """
    One training epoch.

    Concept — Gradient clipping:
      LSTMs can suffer from exploding gradients when sequences are long or
      gradients accumulate across many time steps. Clipping the global
      gradient norm to max_grad_norm prevents parameter updates from
      becoming destabilisingly large, which is especially important when
      using first-order optimisers like Adam on recurrent networks.
    """
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, lengths)
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping (first-order Adam + explicit norm bound)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n_correct  += (logits.argmax(dim=1) == y).sum().item()
        n_total    += y.size(0)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    """One evaluation pass — returns loss, accuracy, all predictions."""
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, lengths, y in loader:
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        logits = model(x, lengths)
        loss   = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds       = logits.argmax(dim=1)
        n_correct  += (preds == y).sum().item()
        n_total    += y.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return total_loss / n_total, n_correct / n_total, all_preds, all_labels


# ════════════════════════════════════════════════════════════════════════════
# 9. GRADIENT FLOW ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def plot_gradient_flow(model, save_path: str = None):
    """
    Plot mean absolute gradient per named parameter layer after a backward pass.

    Concept: Vanishing gradients are a core problem in deep RNNs.
    Layers near the input receive near-zero gradients, meaning they learn
    slowly or not at all. This plot makes that visible — if early LSTM
    layers have ~0 gradients, it motivates moving to BERT (which uses
    residual connections and self-attention to avoid this entirely).
    """
    ave_grads, max_grads, layers = [], [], []
    for name, param in model.named_parameters():
        if param.grad is not None and 'bias' not in name:
            layers.append(name.replace('lstm.', '').replace('weight_', ''))
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.7), 4))
    ax.bar(range(len(ave_grads)), max_grads, alpha=0.4,
           color='#DD8452', label='Max gradient')
    ax.bar(range(len(ave_grads)), ave_grads, alpha=0.85,
           color='#4C72B0', label='Mean |gradient|')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gradient magnitude')
    ax.set_title('Gradient Flow — BiLSTM\n'
                 '(near-zero early layers → vanishing gradient)')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 10. FULL EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_birnn_experiment(
        version:       int   = 2,
        embedding:     str   = 'glove',      # 'glove' | 'word2vec' | 'random'
        glove_path:    str   = 'data/glove.6B.100d.txt',
        w2v_path:      str   = 'data/GoogleNews-vectors-negative300.bin',
        embedding_dim: int   = 100,
        hidden_dim:    int   = 256,
        n_layers:      int   = 2,
        dropout:       float = 0.3,
        freeze_emb:    bool  = False,
        max_len:       int   = 64,
        batch_size:    int   = 64,
        lr:            float = 1e-3,
        n_epochs:      int   = 10,
        patience:      int   = 3,
        save_dir:      str   = 'results',
) -> dict:
    """
    End-to-end BiLSTM experiment.

    Pipeline:
      1. Load + clean data
      2. spaCy tokenisation
      3. Build vocabulary
      4. Load pre-trained embeddings
      5. Build DataLoaders
      6. Train BiLSTM with Adam + gradient clipping
      7. Early stopping on validation loss
      8. Evaluate + plot learning curves + gradient flow
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f'birnn_sst{version}_{embedding}'

    print(f'\n{"═"*55}')
    print(f'  BiLSTM SST-{version} | emb={embedding} | '
          f'hidden={hidden_dim} | layers={n_layers}')
    print(f'{"═"*55}')

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_texts, train_labels, val_texts, val_labels, label_names = \
        load_sst(version)
    n_classes = len(label_names)

    # ── 2. Tokenise ──────────────────────────────────────────────────────────
    print('\n[1/6] Tokenising with spaCy...')
    train_tokens = spacy_tokenize(train_texts)
    val_tokens   = spacy_tokenize(val_texts)

    # ── 3. Vocabulary ────────────────────────────────────────────────────────
    print('[2/6] Building vocabulary...')
    vocab = Vocabulary(min_freq=2).build(train_tokens)

    # ── 4. Embeddings ────────────────────────────────────────────────────────
    print(f'[3/6] Loading {embedding} embeddings...')
    if embedding == 'glove' and Path(glove_path).exists():
        pretrained = load_glove(glove_path, vocab, embedding_dim)
    elif embedding == 'word2vec' and Path(w2v_path).exists():
        embedding_dim = 300
        pretrained = load_word2vec(w2v_path, vocab, embedding_dim)
    else:
        if embedding != 'random':
            print(f'  ⚠ Embedding file not found — falling back to random init.')
        pretrained = make_random_embeddings(vocab, embedding_dim)

    # ── 5. DataLoaders ───────────────────────────────────────────────────────
    print('[4/6] Building DataLoaders...')
    train_ds = SSTDataset(train_tokens, train_labels, vocab, max_len)
    val_ds   = SSTDataset(val_tokens,   val_labels,   vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # ── 6. Model ─────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        vocab_size     = len(vocab),
        embedding_dim  = embedding_dim,
        hidden_dim     = hidden_dim,
        n_layers       = n_layers,
        n_classes      = n_classes,
        dropout        = dropout,
        pretrained_emb = pretrained,
        freeze_emb     = freeze_emb,
    ).to(DEVICE)

    print(f'  Trainable parameters: {model.count_parameters():,}')

    # Class weights for imbalanced SST-5
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        1.0 / (class_counts / class_counts.sum()), dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 7. Optimiser: Adam (first-order) ─────────────────────────────────────
    # Concept: Adam adapts the learning rate per parameter using estimates of
    # first and second moments of the gradient. It converges faster than SGD
    # on sparse gradients (common in NLP) without requiring learning rate tuning.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )

    # ── 8. Training loop with early stopping ─────────────────────────────────
    print(f'\n[5/6] Training ({n_epochs} epochs, patience={patience})...')
    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, _, _ = evaluate_epoch(model, val_loader, criterion)
        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        print(f'  Epoch {epoch:2d}/{n_epochs}  '
              f'train_loss={tr_loss:.4f} acc={tr_acc:.4f}  '
              f'val_loss={va_loss:.4f} acc={va_acc:.4f}  '
              f'({time.time()-t0:.1f}s)')

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'  Early stopping at epoch {epoch}.')
                break

    # ── 9. Restore best checkpoint and evaluate ───────────────────────────────
    model.load_state_dict(best_state)
    print('\n[6/6] Final evaluation...')
    _, _, y_pred, y_true = evaluate_epoch(model, val_loader, criterion)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f'\n  Accuracy : {acc:.4f} ({acc*100:.2f}%)')
    print(f'  Macro-F1 : {mf1:.4f}')
    print(f'\n{classification_report(y_true, y_pred, target_names=label_names)}')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(5, n_classes * 1.4), max(4, n_classes * 1.2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix — BiLSTM SST-{version} ({embedding})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_{tag}.png', dpi=150)
    plt.show()

    # Learning curves
    epochs_ran = len(history['train_loss'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(range(1, epochs_ran+1), history['train_loss'],
             'o-', label='Train', color='#4C72B0')
    ax1.plot(range(1, epochs_ran+1), history['val_loss'],
             's--', label='Val',  color='#DD8452')
    ax1.set_title(f'Loss — BiLSTM SST-{version} ({embedding})')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(range(1, epochs_ran+1), history['train_acc'],
             'o-', label='Train', color='#4C72B0')
    ax2.plot(range(1, epochs_ran+1), history['val_acc'],
             's--', label='Val',  color='#DD8452')
    ax2.set_title(f'Accuracy — BiLSTM SST-{version} ({embedding})')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curve_{tag}.png', dpi=150)
    plt.show()

    # Gradient flow (one backward pass on a batch)
    model.train()
    sample_x, sample_l, sample_y = next(iter(train_loader))
    loss = criterion(model(sample_x.to(DEVICE), sample_l.to(DEVICE)),
                     sample_y.to(DEVICE))
    loss.backward()
    plot_gradient_flow(model,
                       save_path=f'{save_dir}/gradient_flow_{tag}.png')

    return {
        'accuracy':    round(acc, 4),
        'macro_f1':    round(mf1, 4),
        'history':     history,
        'model':       model,
        'vocab':       vocab,
    }
