# Model Choices and Architectural Justification

**Project:** Sentiment Analysis Through the Ages  
**Course:** 7500 — Natural Language Processing | Northeastern University  
**Author:** Michael Dong

---

## Overview

This document explains every major design decision made in this project — why each model was chosen, what alternatives were considered, and what NLP concept each choice demonstrates. It is structured to mirror the three-tier progression of the project.

---

## 1. Why a Three-Tier Progression?

The core analytical claim of this project is that architectural complexity correlates with sentiment classification performance — but not uniformly. The three tiers are:

| Tier | Model | Key limitation addressed |
|---|---|---|
| 1 | N-gram TF-IDF + Logistic Regression | No baseline at all |
| 2 | BiLSTM + Pre-trained Embeddings | Word order and context ignored by Tier 1 |
| 3 | BERT fine-tuned | Fixed vocabulary, sequential processing, no pre-training in Tier 2 |

Keeping the dataset fixed (SST-2 and SST-5) across all tiers means performance differences are attributable only to the model architecture, not to data differences. This is a deliberate controlled-variable design.

---

## 2. Dataset Selection — Why SST?

**Chosen:** Stanford Sentiment Treebank (SST-2 and SST-5)  
**Alternatives considered:** IMDb, Yelp Polarity, Twitter Sentiment140

SST was chosen for four reasons:

**1. Two difficulty levels in one dataset.** SST-2 (binary) and SST-5 (five-class) share the same source sentences. The move from SST-2 to SST-5 is a controlled stress test — the only variable changing is label granularity. This produces a natural "degradation analysis" that reveals each model's representational limits.

**2. Clean, short sentences.** SST sentences are extracted parse-tree fragments (median ~10 words), meaning tokenisation choices have a larger proportional effect than on long-document datasets. This makes Tier 1 vs Tier 2 tokenisation comparisons more analytically visible.

**3. Benchmark saturation.** Published accuracy numbers for SST-2 and SST-5 are extensively documented across decades of NLP literature. Every result in this project can be positioned against a known baseline, making evaluation claims credible.

**4. HuggingFace availability.** Both datasets load in a single line with no preprocessing required, keeping the project focused on model development rather than data engineering.

**Why not IMDb?** IMDb reviews are long (many hundreds of words) and binary only. The long-document setting would make the BiLSTM's max_seq_len=64 design choice harder to justify and would dominate training time on Colab.

**Why not Twitter?** Noisy tokenisation (hashtags, mentions, emoji) would make Tier 1 tokenisation choices the dominant source of variance, obscuring the model comparison.

---

## 3. Tier 1 — N-gram Baseline

### 3.1 Tokenisation: Regex whitespace split

**Why:** The simplest possible tokeniser — `re.findall(r"[a-z']+", text.lower())`. This is intentionally naive. Its known weaknesses (losing punctuation signal, collapsing contractions) establish a clear baseline for what improved tokenisation in Tier 2 contributes.

**Alternative considered:** spaCy (used in Tier 2). Deliberately deferred to create a visible upgrade.

### 3.2 Representation: TF-IDF with unigrams and bigrams

**Why TF-IDF over raw counts:** Raw term frequency overweights very common words. TF-IDF (with `sublinear_tf=True`, applying log(1+tf)) dampens high-frequency terms and upweights discriminative rare terms, which is critical for sentiment where a word like "masterpiece" (rare but strongly positive) should outweigh "the" (common but neutral).

**Why bigrams:** Unigrams alone miss negation patterns. "not good" has opposite sentiment to "good". The bigram `not_good` carries this correctly. The N-gram ablation (notebook 02, Section 8) quantifies this contribution.

**Why max_features=20,000:** SST's vocabulary after frequency pruning (min_df=2) is approximately 14,000 tokens for SST-2. 20,000 is generous enough to include most informative bigrams without becoming computationally expensive.

### 3.3 PoS Features from spaCy (HMM-based tagger)

**Why:** Adding a 17-dimensional PoS tag frequency vector provides syntactic signal that pure bag-of-words misses. Adjectives (ADJ) are the strongest lexical sentiment carriers; their relative frequency differs between positive and negative sentences (shown in EDA notebook Section 6b). spaCy's tagger uses a statistical model trained with Viterbi decoding over an HMM-like CRF, directly demonstrating the HMM concept on the project's own data.

**Why normalised frequencies rather than raw counts:** Sentences vary in length. Normalising to relative frequency makes the PoS features length-invariant.

### 3.4 Optimiser: L-BFGS (second-order)

**Why L-BFGS over SGD:** Logistic regression has a **convex** loss surface (cross-entropy + L2 regularisation). On convex problems, L-BFGS — a quasi-Newton method that approximates the inverse Hessian — achieves superlinear convergence and typically converges in 50–200 iterations rather than the thousands required by first-order SGD. For a fixed dataset of ~67,000 sentences with 20,017 features, this is computationally superior.

**Why not L-BFGS for the neural tiers:** The loss surface of a neural network is non-convex and high-dimensional. Computing even an approximate Hessian for millions of parameters is prohibitively expensive. L-BFGS is used exclusively in Tier 1, where its convexity assumption holds. This contrast between Tiers 1 and 2 directly demonstrates the first-order vs second-order optimisation concept.

---

## 4. Tier 2 — Bidirectional LSTM

### 4.1 Tokenisation: spaCy

**Upgrade from Tier 1:** spaCy's rule-based tokeniser correctly handles punctuation attached to words (`"good,"` → `["good", ","]`) and hyphenated compounds. For SST, the practical difference is modest, but it is an explicit design improvement that the project documents and motivates.

### 4.2 Embeddings: GloVe and Word2Vec

**Why pre-trained embeddings over random initialisation:** SST-2 has ~67,000 training sentences — enough to train a reasonable classifier but not enough to learn reliable word representations from scratch. Pre-trained embeddings initialise the model at a meaningful point in the representation space, reducing the amount of fine-tuning required. The embedding ablation (notebook 03, Section 9) quantifies the gain over random initialisation.

**GloVe (chosen primary):** GloVe (Pennington et al., 2014) trains by factorising a global word-word co-occurrence matrix. It explicitly incorporates corpus-wide statistics, producing embeddings that encode not just semantic similarity but also the ratio of co-occurrence probabilities. For sentiment, the global statistics are informative — "brilliant" and "outstanding" co-occur with similar words across the whole corpus, and GloVe captures this.

**Word2Vec (ablation):** Word2Vec (Mikolov et al., 2013) uses a local context window (Skip-gram or CBOW). It is faster to train and captures local syntactic regularities well. The 300d Google News vectors are trained on 100B words — more data than GloVe 6B, but on a different domain (news vs mixed text). Whether this domain difference helps or hurts on SST movie reviews is an empirical question answered by the ablation.

**OOV strategy:** Tokens not found in GloVe/Word2Vec are initialised with small random vectors sampled from N(0, 0.01). The near-zero magnitude ensures OOV tokens start near the origin and are distinguishable from trained embeddings (which have much larger norms). The `<PAD>` token is always fixed at exactly zero and excluded from gradient updates via `padding_idx=0`.

**Freeze vs fine-tune:** Embeddings are set `freeze_emb=False` — they are updated during training. For SST's sentiment domain, allowing the embeddings to shift toward task-relevant representations improves performance. Freezing would be appropriate if the dataset were very small or if we wanted to preserve general-purpose embeddings for transfer to other tasks.

### 4.3 Architecture: BiLSTM with max-over-time pooling

**Why bidirectional:** A forward LSTM at position t has seen tokens 1…t. A backward LSTM at position t has seen tokens t…T. Concatenating the two hidden states at every position gives the model access to full left and right context simultaneously. For sentiment, this is critical: negation (e.g. "wasn't terrible") requires knowing that "wasn't" precedes "terrible", which the forward pass captures, but also that "either" follows (weakening the negative), which only the backward pass sees.

**Why LSTM over vanilla RNN:** Standard RNNs suffer from vanishing gradients across long sequences — gradients decay exponentially as they are backpropagated through time steps. LSTMs address this with an explicit memory cell and three gates (input, forget, output) that learn when to retain, update, or discard information. The forget gate bias is initialised to 1.0 in this implementation, following Jozefowicz et al. (2015), which encourages the model to retain information by default.

**Why max-over-time pooling over last hidden state:** Using only the last hidden state is a common simplification but discards the intermediate states, which may contain the most sentiment-relevant features (e.g. a strongly negative adjective early in the sentence). Max-over-time pooling takes the element-wise maximum across all time steps, retaining the strongest activated feature from any position. This was shown to outperform last-hidden-state pooling on text classification tasks by Kim (2014).

**Why 2 LSTM layers:** A single-layer BiLSTM learns word-level representations. A second layer operates on top of those representations, learning higher-level sequential abstractions. Two layers is a standard choice that balances capacity with training stability; deeper LSTMs offer diminishing returns on short texts and require careful regularisation.

**Dropout:** Applied to the input embeddings (before LSTM) and to the pooled output (before the classifier). Dropout at the embedding layer acts as a form of word dropout, encouraging the model not to over-rely on single words. Between LSTM layers, dropout is controlled by PyTorch's built-in `dropout` parameter.

### 4.4 Optimiser: Adam (first-order)

**Why Adam over SGD:** Adam (Kingma and Ba, 2015) maintains per-parameter adaptive learning rates using estimates of the first moment (mean) and second moment (uncentred variance) of the gradient. On sparse gradient problems — common in NLP where many embedding rows receive no gradient on a given batch — Adam adapts the learning rate for rarely-updated parameters, while SGD with a global learning rate would update them too slowly or not at all.

**Why not L-BFGS (as in Tier 1):** The non-convex, high-dimensional loss surface of a neural network makes Hessian approximation computationally intractable. Even low-rank Hessian approximations require many forward and backward passes per update step, which is prohibitive for 67,000 training examples. Adam updates in O(parameters) per step.

**Gradient clipping (max_norm=5.0):** LSTM gradients can explode when sequences are long or the training signal is strong. Clipping the global gradient norm to 5.0 bounds the step size independently of the loss magnitude. This is a necessary safeguard specifically for recurrent networks — it is not needed in the logistic regression baseline (convex, no recurrence) or for BERT (gradient clipping there is milder at 1.0).

**Learning rate scheduler:** `ReduceLROnPlateau` monitors validation loss and halves the learning rate when it fails to improve for 2 consecutive epochs. This allows the model to make large updates early in training and fine-grained updates later, without requiring manual learning rate tuning.

---

## 5. Tier 3 — BERT Fine-Tuning

### 5.1 Tokenisation: WordPiece

**Why different cleaning for BERT:** Tiers 1 and 2 normalise PTB-style contractions ("do n't" → "don't") and bracket tokens ("-lrb-" → "("). BERT is pre-trained on text that includes the same PTB conventions — changing them at fine-tuning time creates a distribution mismatch between pre-training and fine-tuning. Therefore, `clean_text_bert()` applies only whitespace stripping.

**Why WordPiece over word-level tokenisation:** WordPiece (Schuster and Nakajima, 2012) builds a vocabulary of subword units by greedily merging frequent character sequences. Rare or unseen words are decomposed into known subwords (e.g. "unbelievable" → ["un", "##be", "##liev", "##able"]). This eliminates the OOV problem entirely — every token in any SST sentence maps to at least one subword piece in BERT's 30,522-token vocabulary.

**[CLS] and [SEP] tokens:** Every BERT input begins with [CLS] (index 101) and ends with [SEP] (index 102). The final hidden state of [CLS] — a learned aggregate representation of the entire sequence — is used as the input to the classification head. This is BERT's intended interface for single-sentence classification tasks.

### 5.2 Model: bert-base-uncased

**Why bert-base-uncased over bert-large or cased:** `bert-base-uncased` (12 layers, 768 hidden dimensions, 12 attention heads, 110M parameters) is the standard choice for fine-tuning experiments on tasks where casing is not semantically critical (sentiment classification is case-insensitive — "GREAT" and "great" carry the same sentiment). `bert-large` (340M parameters) would require more GPU memory and training time, with marginal gains on SST that are unlikely to justify the cost in a coursework setting.

**Why output_attentions=True:** Returning attention weights enables the attention visualisation analysis in notebook 04 (Section 6). Attention heatmaps show which tokens BERT attends to when encoding each position — an interpretability capability that neither TF-IDF weights (Tier 1's feature analysis) nor gradient flow (Tier 2's diagnostic) provides in the same form.

### 5.3 Classification head

A single linear layer maps the 768-dimensional [CLS] representation to n_classes logits. This is the minimal possible head — no additional hidden layers. The reasoning: BERT's 12 transformer layers already produce a rich, task-adaptable representation. Adding more layers on top risks overfitting on SST's small fine-tuning set (67,000 sentences for SST-2, only 8,544 for SST-5) and can interfere with the delicate optimisation balance required for fine-tuning pre-trained weights.

### 5.4 Optimiser: AdamW with linear warmup

**Why AdamW over Adam:** Standard Adam applies L2 regularisation (weight decay) by adding λw to the gradient before the adaptive update. This is incorrect — the adaptive learning rate scales the weight decay differently for each parameter, so effectively-decayed weights depend on the gradient history, not just the weight magnitude. AdamW (Loshchilov and Hutter, 2019) decouples weight decay from the gradient update, applying it directly: w ← w(1-λ) - α·g. This is the theoretically correct form and has been shown empirically to improve generalisation for fine-tuning transformer models.

**Why linear warmup:** BERT's pre-trained weights are in a carefully learned configuration. During the first steps of fine-tuning, the classification head's weights are random, producing large gradients. A high learning rate at this point can destabilise BERT's pre-trained representations — a phenomenon called "catastrophic forgetting". Linear warmup starts the learning rate at 0 and ramps it up to 2e-5 over the first 10% of training steps, giving the classifier head time to stabilise before the pre-trained weights are updated aggressively.

**Why learning rate 2e-5:** The recommended range for BERT fine-tuning from Devlin et al. (2019) is 2e-5 to 5e-5. Lower values (2e-5) are conservative and reduce the risk of catastrophic forgetting, which is the primary concern on a small dataset like SST-5.

**Why 3 epochs:** Devlin et al. (2019) recommend 2–4 epochs for most BERT fine-tuning tasks. More epochs risk overfitting on SST's small training sets, particularly SST-5 (8,544 sentences). The validation loss is monitored with early stopping (patience=2) to halt training if the model begins to overfit before epoch 3.

### 5.5 Why BERT outperforms BiLSTM on SST-5

Three structural advantages combine on the fine-grained task:

1. **Subword tokenisation eliminates OOV.** Sentiment-bearing rare words ("dispiriting", "iridescent") that fall outside GloVe/Word2Vec vocabulary receive meaningful subword representations in BERT.

2. **Bidirectional self-attention captures arbitrary-distance dependencies.** The [CLS] token's representation is computed by attending over all pairs of tokens simultaneously, at every layer. A BiLSTM's bidirectionality only captures left and right context independently; self-attention captures all pairwise relationships jointly.

3. **Pre-training on 3.3 billion words.** BERT has seen far more linguistic diversity than SST can provide for fine-tuning. Its representations encode nuanced semantic and syntactic structure that makes fine-grained distinctions (e.g. "merely adequate" vs "genuinely good") more tractable.

---

## 6. Evaluation Design

### Why accuracy AND macro-F1?

Accuracy alone is a misleading metric when class distributions are imbalanced. SST-5's classes are not uniform — "neutral" is underrepresented. A model that simply predicts "negative" for all ambiguous cases will achieve reasonable accuracy but poor macro-F1 (which weights each class equally regardless of support). Reporting both metrics exposes this behaviour.

### Why confusion matrices?

Confusion matrices reveal *which* classes are confused, not just how often. For SST-5, the expected finding is that adjacent classes ("negative" vs "very negative", "positive" vs "very positive") are the primary error sources — because they share similar vocabulary and the distinction is one of intensity rather than polarity. This is a qualitative insight that summary metrics miss.

### Why the SST-2 → SST-5 degradation analysis?

The degradation from binary to fine-grained classification is the central analytical claim of the project: simpler models degrade more. This is quantified by comparing the accuracy drop for each tier. It directly motivates the architectural progression — if all three tiers degraded equally, the project's narrative would fail.

### Why learning curves?

Learning curves (training and validation loss/accuracy vs. epoch) diagnose overfitting and underfitting. For BiRNN and BERT, a growing gap between training and validation curves indicates overfitting; a slow decline in both indicates underfitting or a learning rate that is too low. The curves also reveal whether early stopping triggered at the right time.

---

## 7. What Was Explicitly Not Done (and Why)

| Choice | Reason for exclusion |
|---|---|
| bert-large | 340M parameters; requires more GPU than a free Colab instance provides reliably |
| RoBERTa / ALBERT | Interesting extensions but would require duplicating the full fine-tuning pipeline; noted as future work in the README |
| LSTM encoder-decoder (seq2seq) | Relevant for NMT (Project 3) but not for classification |
| Attention mechanism on top of BiLSTM | Would blur the architectural boundary between Tier 2 and Tier 3; BiLSTM is intentionally kept as a clean baseline |
| Cross-validation | SST's fixed train/val/test splits are the standard benchmark protocol; deviating would make comparison with literature harder |
| Hyperparameter search | Exhaustive tuning would consume Colab compute; literature-informed defaults are used instead, cited to their sources |
