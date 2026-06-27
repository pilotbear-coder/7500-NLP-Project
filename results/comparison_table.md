# Results: Three-Tier Comparison Table

**Project:** Sentiment Analysis Through the Ages  
**Dataset:** Stanford Sentiment Treebank — SST-2 (binary) and SST-5 (fine-grained)  
**Evaluation split:** Validation set (SST-2: 872 sentences | SST-5: 1,101 sentences)

> This file is the canonical results reference for the project.  
> All numbers are filled in after running notebooks 02, 03, and 04 in order.  
> Placeholders marked `—` are replaced with your actual run results.

---

## Primary Results Table

| Model | SST-2 Accuracy | SST-2 Macro-F1 | SST-5 Accuracy | SST-5 Macro-F1 | Train Time (SST-2) |
|---|---|---|---|---|---|
| **Tier 1 — N-gram + L-BFGS** | — | — | — | — | — |
| **Tier 2 — BiLSTM + GloVe** | — | — | — | — | — |
| **Tier 2 — BiLSTM + Word2Vec** | — | — | — | — | — |
| **Tier 3 — BERT fine-tuned** | — | — | — | — | — |

---

## Literature Benchmarks (for reference)

| Model | SST-2 Accuracy | SST-5 Accuracy | Source |
|---|---|---|---|
| Naive Bayes (bigram) | 83.1% | 41.0% | Socher et al. (2013) |
| SVM | 79.4% | 40.7% | Socher et al. (2013) |
| BiLSTM + GloVe | ~88% | ~46% | Sun et al. (2025) |
| BERT-base fine-tuned | ~93% | ~53% | Munikar et al. (2019) |
| BERT + BiLSTM | 97.7% | 59.5% | Nkhata et al. (2025) |

---

## SST-2 → SST-5 Degradation

How much does each model suffer when moving from binary to 5-class classification?  
Larger drops reveal the limits of that architecture's representations.

| Model | Accuracy Drop (pp) | Macro-F1 Drop (pp) |
|---|---|---|
| Tier 1 — N-gram + L-BFGS | — | — |
| Tier 2 — BiLSTM + GloVe | — | — |
| Tier 3 — BERT fine-tuned | — | — |

**Expected pattern:** Tier 1 suffers the largest drop (bag-of-words cannot distinguish "mildly negative" from "very negative"). Tier 3 suffers the least, because BERT's contextual representations encode fine-grained distinctions that intensifiers and negation scope produce.

---

## Embedding Ablation (Tier 2, SST-2 only)

All hyperparameters held constant — only the embedding initialisation changes.

| Embedding | Dim | Accuracy | Macro-F1 | OOV Strategy |
|---|---|---|---|---|
| Random (no pre-training) | 100d | — | — | Random N(0, 0.01) |
| GloVe 6B | 100d | — | — | Random N(0, 0.01) |
| Word2Vec Google News | 300d | — | — | Random N(0, 0.01) |

**Expected pattern:** Pre-trained embeddings (GloVe, Word2Vec) outperform random initialisation, especially on the smaller SST-5 training set where the model has less data to learn meaningful representations from scratch.

---

## N-gram Range Ablation (Tier 1, SST-2)

| N-gram Range | Vocab Size | Accuracy | Macro-F1 |
|---|---|---|---|
| Unigram only (1,1) | — | — | — |
| Uni + Bigram (1,2) | — | — | — |
| Uni + Bi + Trigram (1,3) | — | — | — |

**Expected pattern:** Bigrams add meaningful signal (capturing "not good", "very bad"). Trigrams often add noise on a corpus as clean as SST.

---

## Per-Class Performance (SST-5 validation set)

### Tier 1 — N-gram Baseline

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| very negative | — | — | — | — |
| negative | — | — | — | — |
| neutral | — | — | — | — |
| positive | — | — | — | — |
| very positive | — | — | — | — |

### Tier 2 — BiLSTM + GloVe

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| very negative | — | — | — | — |
| negative | — | — | — | — |
| neutral | — | — | — | — |
| positive | — | — | — | — |
| very positive | — | — | — | — |

### Tier 3 — BERT

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| very negative | — | — | — | — |
| negative | — | — | — | — |
| neutral | — | — | — | — |
| positive | — | — | — | — |
| very positive | — | — | — | — |

**Expected pattern:** The `neutral` class is consistently the hardest across all tiers. It is the smallest class and sits between positive and negative, so models tend to misclassify it as one of its neighbours.

---

## Hyperparameters Used

### Tier 1 — N-gram Baseline (`models/baseline.py`)

| Parameter | Value | Justification |
|---|---|---|
| N-gram range | (1, 2) | Bigrams capture negation phrases ("not good") |
| Max features | 20,000 | Covers SST vocabulary with frequency cutoff |
| Sublinear TF | True | Dampens effect of very high-frequency terms |
| Min document freq | 2 | Removes hapax legomena |
| Optimiser | L-BFGS | Second-order; optimal for convex logistic loss |
| Regularisation C | 1.0 | Default; tunable via cross-validation |
| Class weights | Balanced | Corrects mild SST-5 class imbalance |

### Tier 2 — BiLSTM (`models/birnn.py`)

| Parameter | Value | Justification |
|---|---|---|
| Embedding dim | 100d (GloVe) / 300d (Word2Vec) | Standard pre-trained sizes |
| Hidden dim | 256 | Balances capacity and compute |
| LSTM layers | 2 | Deeper representation; second layer learns higher abstractions |
| Dropout | 0.3 | Applied to embeddings and pre-classifier |
| Max seq length | 64 | Covers >95% of SST sentences (median ~10 words) |
| Batch size | 64 | Standard for RNNs on this scale |
| Optimiser | Adam | First-order adaptive; standard for deep learning |
| Learning rate | 1e-3 | Adam default; works well with pre-trained embeddings |
| Weight decay | 1e-5 | Light L2 regularisation |
| Gradient clipping | 5.0 | Prevents exploding gradients in LSTM |
| Patience | 3 epochs | Early stopping on validation loss |
| Embedding freeze | False | Allow embeddings to adapt to sentiment domain |

### Tier 3 — BERT (`models/bert_finetune.py