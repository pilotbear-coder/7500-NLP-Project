# Sentiment Analysis Through the Ages: N-gram Baseline → BiRNN → BERT on SST-2/SST-5

**Course:** NLP 7500 — Natural Language Processing | Northeastern University  
**Author:** Michael Dong (`dong.mic@northeastern.edu`) | Group 14

---

## Project Overview

This project implements and compares three generations of NLP architecture for sentence-level sentiment classification, using a fixed dataset (Stanford Sentiment Treebank) to isolate architectural improvements as the sole variable. The progression — statistical baseline → deep recurrent model → pre-trained Transformer — mirrors the historical development of the field and allows direct, controlled benchmarking at each stage.

**Exact task:** Given a single English sentence drawn from a movie review, predict its sentiment label. In the binary setting (SST-2), the label is either *positive* (1) or *negative* (0). In the fine-grained setting (SST-5), the label is one of five classes: *very negative* (0), *negative* (1), *neutral* (2), *positive* (3), or *very positive* (4).

**Core research question:** How much does architectural sophistication — specifically the shift from surface-level n-gram statistics to contextual BiRNN representations to full Transformer pre-training — improve sentiment classification accuracy on the same dataset, and at what computational cost?

---

## Problem Statement

Sentiment analysis at the binary level (positive vs. negative) is a largely solved problem for clean, sentence-length text. The harder and more practically relevant challenge is *fine-grained* sentiment classification, where a model must distinguish between shades of sentiment (e.g., "mildly negative" vs. "very negative") that often hinge on negation, irony, intensifiers, and subtle compositional structure. Simple bag-of-words models collapse these distinctions; contextual models theoretically preserve them.

This project operationalises that gap concretely:

1. We establish a reproducible N-gram/HMM baseline with known performance on SST-2 (~84% accuracy from literature).
2. We train a BiRNN model with pre-trained word embeddings (GloVe and Word2Vec) on the same data and measure the gain.
3. We fine-tune BERT-base on SST-2, then extend all three models to SST-5 to expose how each architecture handles the jump to 5-class classification.
4. We analyse *why* each generation improves (or fails to improve), tying results back to core NLP concepts: tokenization, gradient flow, optimization, and contextual representation.

---

## Dataset

### Source
Stanford Sentiment Treebank (SST), introduced by Socher et al. (2013) at EMNLP. Available via HuggingFace Datasets with a single line of code.

```python
from datasets import load_dataset
sst2 = load_dataset("stanfordnlp/sst2")
sst5 = load_dataset("SetFit/sst5")
```

### Structure and Volume

| Split | SST-2 sentences | SST-5 sentences |
|---|---|---|
| Train | 67,349 | 8,544 |
| Validation | 872 | 1,101 |
| Test | 1,821 (labels hidden) | 2,210 |

SST was constructed from 11,855 sentences extracted from Rotten Tomatoes movie reviews, parsed with the Stanford Parser. Every syntactic phrase in each parse tree was independently annotated for sentiment by Amazon Mechanical Turk workers using a continuous slider (converted to discrete labels). This gives SST a unique property: phrase-level annotations, not just sentence-level, which allows analysis of compositional sentiment.

### Labels

**SST-2 (binary):**
- `0` — Negative (includes "somewhat negative")
- `1` — Positive (includes "somewhat positive")
- Neutral sentences are discarded in SST-2.

**SST-5 (fine-grained):**
- `0` — Very Negative
- `1` — Negative
- `2` — Neutral
- `3` — Positive
- `4` — Very Positive

### Class Distribution (SST-2 Training Set)
- Negative: ~44.2% (29,780 sentences)
- Positive: ~55.8% (37,569 sentences)
- Mild class imbalance; addressed by reporting macro-F1 alongside accuracy.

### Preprocessing Pipeline

| Step | N-gram Baseline | BiRNN | BERT |
|---|---|---|---|
| Lowercasing | Yes | Yes | No (BERT is case-sensitive by default) |
| Tokenization | Whitespace + punctuation split | SpaCy tokenizer | WordPiece (bert-base-uncased tokenizer) |
| Vocabulary | Top 20,000 tokens by frequency | Pre-trained GloVe/Word2Vec vocab | BERT vocab (30,522 tokens) |
| OOV handling | `<UNK>` token | Pre-trained embedding lookup; random init for OOV | Subword tokenization (no true OOV) |
| Max sequence length | N/A (bag-of-words) | 64 tokens (pad/truncate) | 128 tokens (pad/truncate) |
| Special tokens | None | `<PAD>`, `<UNK>`, `<SOS>`, `<EOS>` | `[CLS]`, `[SEP]`, `[PAD]` |

---

## Models

### Tier 1 — N-gram / HMM Baseline
- **TF-IDF Vectorizer** (unigrams + bigrams, top 20k features) + **Logistic Regression**
- **HMM-based sequence tagger** for PoS-aware feature engineering
- Optimizer: L-BFGS (second-order) for logistic regression; Baum-Welch (EM) for HMM
- Target performance: ~84% accuracy on SST-2 (Socher et al., 2013 NB/SVM baselines)

### Tier 2 — Bidirectional RNN with Pre-trained Embeddings
- **BiLSTM** (2 layers, hidden size 256, dropout 0.3) with a linear classification head
- Embeddings: GloVe 100d and Word2Vec 300d (Google News), compared separately
- Optimizer: Adam (first-order, lr=1e-3) with gradient clipping (max norm 5.0)
- Target performance: ~88–90% accuracy on SST-2

### Tier 3 — BERT Fine-Tuning
- **bert-base-uncased** (12 layers, 768 hidden, 110M parameters) with a classification head on the `[CLS]` token
- Optimizer: AdamW (lr=2e-5, weight decay=0.01) with linear warmup schedule
- Fine-tuning: 3 epochs, batch size 32
- Target performance: ~93%+ accuracy on SST-2 (Devlin et al., 2019)

---

## Evaluation Plan

### Primary Metrics
- **Accuracy** — overall percentage of correct predictions
- **Macro-F1** — unweighted mean F1 across all classes; penalises poor performance on minority classes (critical for SST-5)
- **Confusion Matrix** — per-class breakdown to identify which sentiment boundaries each model struggles with

### Baseline Anchors (from literature)

| Model | SST-2 Accuracy | SST-5 Accuracy | Source |
|---|---|---|---|
| Naive Bayes (bigram) | 83.1% | 41.0% | Socher et al. (2013) |
| SVM | 79.4% | 40.7% | Socher et al. (2013) |
| BiLSTM + GloVe | ~88% | ~46% | Sun et al. (2025) |
| BERT-base fine-tuned | ~93% | ~53% | Munikar et al. (2019) |
| BERT + BiLSTM | ~97.7% | ~59.5% | Nkhata et al. (2025) |

### Comparison Protocol
All three models are evaluated on the **same validation split** of SST-2 and SST-5. No model sees test labels during training. Results are reported in a single comparison table (see `results/comparison_table.md`).

### Additional Analyses
- **Learning curves:** Training and validation loss/accuracy vs. epoch for BiRNN and BERT, to diagnose overfitting.
- **Gradient flow analysis:** Visualisation of gradient norms per layer in the BiRNN to motivate the shift to BERT.
- **Embedding comparison:** GloVe vs. Word2Vec ablation in Tier 2, held constant across all other hyperparameters.
- **SST-2 → SST-5 degradation:** For each model, measure the accuracy drop moving from binary to 5-class. This is the core analytical finding of the project.
- **Error analysis:** Sample 50 misclassified examples per model and categorise error types (negation, intensifiers, neutral sentences).

---

## Core NLP Concepts Covered

| Concept | Where it appears |
|---|---|
| Tokenization | Compared across all three tiers (whitespace vs. SpaCy vs. WordPiece) |
| N-grams & HMM | Tier 1 baseline implementation |
| Word2Vec / GloVe | Tier 2 embedding initialisation and ablation |
| Recurrent Neural Networks | BiLSTM architecture in Tier 2 |
| Bidirectional RNN | Core architecture of Tier 2 |
| Gradient Descent | Adam (Tier 2), AdamW (Tier 3), L-BFGS (Tier 1) |
| First/Second-Order Optimization | L-BFGS (second-order) vs. Adam (first-order) comparison |
| PoS Tagging | Feature engineering for HMM baseline |
| Transformers / BERT | Tier 3 fine-tuning |
| Evaluation Techniques | Accuracy, macro-F1, confusion matrix, learning curves |
| Sentiment Analysis | The primary task throughout |

---

## Repository Structure

```
sentiment-analysis-evolution/
├── README.md
├── requirements.txt
├── data/
│   └── README.md                  # Dataset description; no raw data stored
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis of SST-2/SST-5
│   ├── 02_baseline.ipynb          # N-gram TF-IDF + HMM baseline
│   ├── 03_birnn.ipynb             # BiRNN with GloVe and Word2Vec
│   └── 04_bert.ipynb              # BERT fine-tuning
├── models/
│   ├── baseline.py                # TF-IDF + Logistic Regression pipeline
│   ├── birnn.py                   # BiLSTM model definition (PyTorch)
│   └── bert_finetune.py           # BERT fine-tuning with HuggingFace
├── evaluation/
│   └── metrics.py                 # Shared evaluation functions
├── results/
│   └── comparison_table.md        # Final results across all models and datasets
└── docs/
    └── model_choices.md           # Justification of architectural decisions
```

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentiment-analysis-evolution.git
cd sentiment-analysis-evolution

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** includes: `torch`, `transformers`, `datasets`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `spacy`, `gensim`, `jupyter`

---

## References

1. Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. *EMNLP 2013*, 1631–1642. https://aclanthology.org/D13-1170/

2. Munikar, M., Shakya, S., & Shrestha, A. (2019). Fine-grained sentiment classification using BERT. *arXiv:1910.03474*. https://arxiv.org/abs/1910.03474

3. Sun, Y., et al. (2025). Sentiment analysis using long short term memory and amended dwarf mongoose optimization algorithm. *Scientific Reports*. https://www.nature.com/articles/s41598-025-01834-1

4. Nkhata, G., et al. (2025). Fine-tuning BERT with Bidirectional LSTM for fine-grained movie reviews sentiment analysis. *arXiv:2502.20682*. https://arxiv.org/abs/2502.20682

5. Cheang, B., Wei, B., Kogan, D., Qiu, H., & Ahmed, M. (2020). Language representation models for fine-grained sentiment classification. *arXiv:2005.13619*. https://arxiv.org/abs/2005.13619

6. Bansidhar, P. M. (2021). Distilling BERT for low complexity network training. *arXiv:2105.06514*. https://arxiv.org/abs/2105.06514

7. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. https://aclanthology.org/N19-1423/
