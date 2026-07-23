"""
bert_finetune.py
================
Tier 3 — BERT Fine-Tuning for Sentiment Classification
Dataset: SST-2 (binary) and SST-5 (fine-grained)

Concepts demonstrated:
  - Transformers / BERT architecture
  - WordPiece tokenisation ([CLS], [SEP], attention masks)
  - Transfer learning / fine-tuning
  - AdamW optimiser with linear warmup (first-order)
  - Attention weight visualisation (what BERT attends to)
  - Evaluation: accuracy, macro-F1, confusion matrix, learning curves
  - SST-2 → SST-5 degradation compared to baseline and BiRNN

Usage:
    from models.bert_finetune import BERTClassifier, run_bert_experiment
    results = run_bert_experiment(version=2)
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertModel,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ── Confirmed dataset field names ────────────────────────────────────────────
# stanfordnlp/sst2 → 'sentence' | SetFit/sst5 → 'text'

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'bert-base-uncased'
print(f'Device: {DEVICE}  |  BERT model: {MODEL_NAME}')


# ════════════════════════════════════════════════════════════════════════════
# 1. TEXT CLEANING  (BERT version — whitespace strip only)
# ════════════════════════════════════════════════════════════════════════════

def clean_text_bert(text: str) -> str:
    """
    Minimal cleaning for BERT input: strip leading/trailing whitespace only.

    Concept: BERT's WordPiece tokeniser was pre-trained on text that
    includes PTB-style conventions (e.g. split contractions "do n't").
    Normalising these AWAY from pre-training conventions can hurt
    performance — so we leave them intact and only strip whitespace.
    This is explicitly different from the cleaning applied in Tiers 1 & 2.
    """
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_sst(version: int = 2):
    """
    Load SST-2 or SST-5 from HuggingFace and apply BERT-appropriate cleaning.

    Returns: train_texts, train_labels, val_texts, val_labels, label_names
    """
    assert version in (2, 5), "version must be 2 or 5"

    if version == 2:
        ds = load_dataset('stanfordnlp/sst2')
        train_texts  = [clean_text_bert(t) for t in ds['train']['sentence']]
        train_labels = list(ds['train']['label'])
        val_texts    = [clean_text_bert(t) for t in ds['validation']['sentence']]
        val_labels   = list(ds['validation']['label'])
        label_names  = ['negative', 'positive']
    else:
        ds = load_dataset('SetFit/sst5')
        train_texts  = [clean_text_bert(t) for t in ds['train']['text']]
        train_labels = list(ds['train']['label'])
        val_texts    = [clean_text_bert(t) for t in ds['validation']['text']]
        val_labels   = list(ds['validation']['label'])
        label_names  = ['very negative', 'negative', 'neutral',
                        'positive', 'very positive']

    print(f'\n── SST-{version} loaded ──────────────────────────────────')
    print(f'  Train : {len(train_texts):,} | Val: {len(val_texts):,}')
    print(f'  Labels: {label_names}')
    return train_texts, train_labels, val_texts, val_labels, label_names


# ════════════════════════════════════════════════════════════════════════════
# 3. WORDPIECE TOKENISATION + DATASET
# ════════════════════════════════════════════════════════════════════════════

class SSTBertDataset(Dataset):
    """
    PyTorch Dataset for BERT fine-tuning on SST.

    Concept — WordPiece tokenisation:
      BERT uses a subword vocabulary of ~30,522 tokens. Unknown words are
      split into known subword pieces (e.g. "unbelievable" → "un", "##be",
      "##liev", "##able"). This means there are effectively NO OOV tokens,
      unlike the fixed vocabulary in Tier 1 and Tier 2.

    Special tokens:
      [CLS] (index 101): prepended to every input. The final hidden state
            of [CLS] is used as the sentence representation for classification.
      [SEP] (index 102): appended after each sentence (and between sentence
            pairs in tasks like NLI or QA; SST uses single sentences).
      [PAD] (index 0):   used to pad sequences to a uniform length within
            each batch. Attention masks prevent the model from attending
            to padding positions.
    """

    def __init__(self, texts: list, labels: list,
                 tokenizer: BertTokenizerFast, max_len: int = 128):
        self.labels = torch.tensor(labels, dtype=torch.long)
        encoded = tokenizer(
            texts,
            padding        = 'max_length',
            truncation     = True,
            max_length     = max_len,
            return_tensors = 'pt',
        )
        self.input_ids      = encoded['input_ids']
        self.attention_mask = encoded['attention_mask']
        self.token_type_ids = encoded.get(
            'token_type_ids',
            torch.zeros_like(encoded['input_ids'])
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'token_type_ids': self.token_type_ids[idx],
            'label':          self.labels[idx],
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. BERT CLASSIFIER MODEL
# ════════════════════════════════════════════════════════════════════════════

class BERTClassifier(nn.Module):
    """
    BERT-base fine-tuned for sequence classification.

    Architecture:
      BERT encoder (12 transformer layers) → [CLS] hidden state
      → Dropout → Linear classification head

    Concept — Transformers / BERT:
      Unlike BiLSTM which processes tokens sequentially, BERT uses
      self-attention to relate every token to every other token in parallel.
      This captures long-range dependencies (e.g. sentiment modifiers far
      from the word they modify) without vanishing gradient issues.

      BERT is pre-trained on two tasks:
        1. Masked Language Modelling (MLM): predict masked tokens
        2. Next Sentence Prediction (NSP): predict if sentence B follows A
      Fine-tuning adds a task-specific head and updates ALL weights end-to-end
      on the downstream task — in our case, sentiment classification.

    Concept — Transfer learning:
      Rather than training from scratch (which requires massive data and
      compute), we initialise from bert-base-uncased weights that encode
      rich linguistic knowledge, then fine-tune on SST for a small number
      of epochs. This is why BERT outperforms BiRNN even on a small dataset.
    """

    def __init__(self, n_classes: int, dropout: float = 0.1,
                 freeze_bert: bool = False):
        super().__init__()
        self.bert    = BertModel.from_pretrained(
            MODEL_NAME, output_attentions=True   # return attn weights for viz
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, n_classes   # 768 → n_classes
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Returns logits and attention weights.
        logits      : (batch, n_classes)
        attentions  : tuple of 12 tensors, each (batch, 12 heads, L, L)
        """
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits     = self.classifier(self.dropout(cls_output))
        return logits, outputs.attentions

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion):
    """
    One BERT fine-tuning epoch.

    Concept — AdamW optimiser:
      Standard Adam applies L2 regularisation incorrectly when combined with
      adaptive learning rates (the weight decay gets absorbed into the moment
      estimates). AdamW fixes this by decoupling weight decay from the
      gradient update, making regularisation more effective for fine-tuning
      large pre-trained models.

    Concept — Linear warmup schedule:
      BERT is sensitive to the learning rate during the first steps of
      fine-tuning. A warmup period linearly increases lr from 0 to the
      target value over the first ~10% of training steps, then linearly
      decays it to 0. This prevents catastrophic forgetting of pre-trained
      weights during the initial high-lr phase.
    """
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(input_ids, attention_mask, token_type_ids)
        loss      = criterion(logits, labels)
        loss.backward()

        # Gradient clipping (recommended for BERT fine-tuning)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        n_correct  += (logits.argmax(dim=1) == labels).sum().item()
        n_total    += labels.size(0)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    """One evaluation pass — returns loss, accuracy, predictions, labels."""
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        logits, _ = model(input_ids, attention_mask, token_type_ids)
        loss      = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=1)
        n_correct  += (preds == labels).sum().item()
        n_total    += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / n_total, n_correct / n_total, all_preds, all_labels


# ════════════════════════════════════════════════════════════════════════════
# 6. ATTENTION VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualise_attention(model, tokenizer, text: str,
                        layer: int = 11, head: int = 0,
                        save_path: str = None):
    """
    Visualise BERT's attention weights for a single sentence.

    Concept: Each of BERT's 12 layers contains 12 attention heads.
    Each head learns a different type of dependency (syntactic, semantic,
    positional). Visualising these shows WHAT the model attends to when
    making a sentiment prediction — a key interpretability tool that
    bag-of-words and even BiRNN lack.

    layer : 0–11 (last layer = 11 is most task-relevant after fine-tuning)
    head  : 0–11
    """
    model.eval()
    tokens_out = tokenizer(text, return_tensors='pt')
    input_ids  = tokens_out['input_ids'].to(DEVICE)
    attn_mask  = tokens_out['attention_mask'].to(DEVICE)
    token_ids  = tokens_out.get(
        'token_type_ids', torch.zeros_like(input_ids)
    ).to(DEVICE)

    _, attentions = model(input_ids, attn_mask, token_ids)
    # attentions: tuple of 12 tensors, each (1, 12, L, L)

    attn = attentions[layer][0, head].cpu().numpy()  # (L, L)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5),
                                    max(5, len(tokens) * 0.45)))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='Blues', ax=ax, cbar=True, vmin=0)
    ax.set_title(f'BERT Attention — Layer {layer+1}, Head {head+1}\n"{text}"',
                 fontsize=10)
    ax.set_xlabel('Key tokens (attended to)')
    ax.set_ylabel('Query tokens (attending from)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 7. FULL EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_bert_experiment(
        version:      int   = 2,
        max_len:      int   = 128,
        batch_size:   int   = 32,
        lr:           float = 2e-5,
        weight_decay: float = 0.01,
        n_epochs:     int   = 3,
        warmup_ratio: float = 0.1,
        dropout:      float = 0.1,
        patience:     int   = 2,
        freeze_bert:  bool  = False,
        save_dir:     str   = 'results',
) -> dict:
    """
    End-to-end BERT fine-tuning experiment.

    Pipeline:
      1. Load + clean data (whitespace strip only for BERT)
      2. WordPiece tokenisation via BertTokenizerFast
      3. Build DataLoaders
      4. Load bert-base-uncased + add classification head
      5. Fine-tune with AdamW + linear warmup schedule
      6. Early stopping on validation loss
      7. Evaluate + confusion matrix + learning curves
      8. Attention visualisation on example sentences
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f'bert_sst{version}'

    print(f'\n{"═"*55}')
    print(f'  BERT Fine-Tuning SST-{version} | lr={lr} | '
          f'epochs={n_epochs} | batch={batch_size}')
    print(f'{"═"*55}')

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_texts, train_labels, val_texts, val_labels, label_names = \
        load_sst(version)
    n_classes = len(label_names)

    # ── 2. Tokeniser ─────────────────────────────────────────────────────────
    print('\n[1/5] Loading BERT tokeniser...')
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f'  Vocab size: {tokenizer.vocab_size:,} | '
          f'max_len: {max_len}')

    # Token length sanity check
    sample_lens = [
        len(tokenizer.encode(t, truncation=False))
        for t in train_texts[:500]
    ]
    print(f'  Token length (sample 500) — '
          f'mean: {np.mean(sample_lens):.1f} | '
          f'max: {max(sample_lens)} | '
          f'>128: {sum(l > 128 for l in sample_lens)}')

    # ── 3. DataLoaders ───────────────────────────────────────────────────────
    print('[2/5] Building datasets and DataLoaders...')
    train_ds = SSTBertDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds   = SSTBertDataset(val_texts,   val_labels,   tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              num_workers=0)

    # ── 4. Model ─────────────────────────────────────────────────────────────
    print('[3/5] Loading BERT model...')
    model = BERTClassifier(
        n_classes   = n_classes,
        dropout     = dropout,
        freeze_bert = freeze_bert,
    ).to(DEVICE)
    print(f'  Trainable parameters: {model.count_parameters():,}')

    # Class weights for SST-5 imbalance
    class_counts  = np.bincount(train_labels)
    class_weights = torch.tensor(
        1.0 / (class_counts / class_counts.sum()), dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 5. Optimiser + scheduler ─────────────────────────────────────────────
    # AdamW: decoupled weight decay (correct for fine-tuning pre-trained models)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps  = len(train_loader) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    print(f'  Total steps: {total_steps} | Warmup steps: {warmup_steps}')

    # ── 6. Training loop ─────────────────────────────────────────────────────
    print(f'\n[4/5] Fine-tuning ({n_epochs} epochs)...')
    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}
    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader,
                                      optimizer, scheduler, criterion)
        va_loss, va_acc, _, _ = evaluate_epoch(model, val_loader, criterion)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        print(f'  Epoch {epoch}/{n_epochs}  '
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

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    print('\n[5/5] Final evaluation...')
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
    plt.title(f'Confusion Matrix — BERT SST-{version}')
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
    ax1.set_title(f'Loss — BERT SST-{version}')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(range(1, epochs_ran+1), history['train_acc'],
             'o-', label='Train', color='#4C72B0')
    ax2.plot(range(1, epochs_ran+1), history['val_acc'],
             's--', label='Val',  color='#DD8452')
    ax2.set_title(f'Accuracy — BERT SST-{version}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curve_{tag}.png', dpi=150)
    plt.show()

    # ── 8. Attention visualisation ───────────────────────────────────────────
    print('\nAttention visualisation on example sentences...')
    examples = [
        "The film is an absolute masterpiece.",
        "I couldn't sit through more than twenty minutes of this.",
        "It's not the worst movie ever made, but it comes close.",
    ]
    for i, ex in enumerate(examples):
        visualise_attention(
            model, tokenizer, ex,
            layer=11, head=0,
            save_path=f'{save_dir}/attention_{tag}_ex{i+1}.png'
        )

    return {
        'accuracy':  round(acc, 4),
        'macro_f1':  round(mf1, 4),
        'history':   history,
        'model':     model,
        'tokenizer': tokenizer,
    }
