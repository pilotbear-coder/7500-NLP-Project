"""
baseline.py
===========
Tier 1 Baseline — N-gram TF-IDF + Logistic Regression + HMM PoS features
Dataset: SST-2 (binary) and SST-5 (fine-grained)

Concepts demonstrated:
  - Tokenization (whitespace + punctuation)
  - N-gram language modelling
  - HMM (for PoS-based feature extraction via hmmlearn)
  - TF-IDF vectorisation
  - Second-order optimisation (L-BFGS via sklearn's lbfgs solver)
  - Evaluation: accuracy, macro-F1, confusion matrix
"""

import re
import time
import numpy as np

# ── Confirmed dataset field names (verified against HuggingFace dataset cards) ──
# stanfordnlp/sst2  → text field: 'sentence' | label: 0/1
# SetFit/sst5       → text field: 'text'     | label: 0-4 | extra: 'label_text'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import spacy

# ── Load spaCy for PoS tagging (used in HMM feature block) ──────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_sst(version: int = 2):
    """
    Load SST-2 or SST-5 from HuggingFace.

    SST-2: stanfordnlp/sst2   — labels: 0 (neg), 1 (pos)
    SST-5: SetFit/sst5         — labels: 0–4 (very neg → very pos)
    """
    assert version in (2, 5), "version must be 2 or 5"

    if version == 2:
        ds = load_dataset("stanfordnlp/sst2")
        label_names = ["negative", "positive"]
    else:
        ds = load_dataset("SetFit/sst5")
        label_names = ["very negative", "negative", "neutral", "positive", "very positive"]

    train_texts  = ds["train"]["sentence"] if version == 2 else ds["train"]["text"]
    train_labels = ds["train"]["label"]
    val_texts    = ds["validation"]["sentence"] if version == 2 else ds["validation"]["text"]
    val_labels   = ds["validation"]["label"]

    print(f"\n── SST-{version} loaded ──────────────────────────────────")
    print(f"  Train : {len(train_texts):,} sentences")
    print(f"  Val   : {len(val_texts):,}  sentences")
    print(f"  Labels: {label_names}")
    return train_texts, train_labels, val_texts, val_labels, label_names


# ════════════════════════════════════════════════════════════════════════════
# 2. TOKENISATION
# ════════════════════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> list[str]:
    """
    Whitespace + punctuation tokeniser.
    Lowercases and strips punctuation — the simplest possible baseline.

    Concept: Tokenization is the first step in any NLP pipeline.
    This deliberately naive tokeniser will be compared against
    SpaCy (Tier 2) and WordPiece (Tier 3/BERT).
    """
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens


def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    return [simple_tokenize(t) for t in texts]


# ════════════════════════════════════════════════════════════════════════════
# 2b. TEXT CLEANING  (run before tokenisation)
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str, for_bert: bool = False) -> str:
    """
    Clean a single SST sentence before tokenisation.

    Issues addressed (discovered in 01_eda.ipynb):
      - Trailing/leading whitespace (common in SST-2 'sentence' field)
      - PTB bracket tokens: -lrb- → (, -rrb- → )
      - Split contractions: "do n't" → "don't", "it 's" → "it's"
      - Multiple internal spaces

    for_bert=True: only strip whitespace — BERT's WordPiece tokeniser
    handles PTB-style text natively (pre-training used the same conventions).
    """
    if for_bert:
        return text.strip()

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
# 3. HMM PoS FEATURE EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_pos_features(texts: list[str], batch_size: int = 512) -> np.ndarray:
    """
    Use spaCy's PoS tagger (which itself uses a statistical model) to extract
    part-of-speech tag distribution as features for each sentence.

    Concept: Hidden Markov Models underlie classical PoS taggers.
    Here we leverage spaCy's tagger to add syntactic signal on top of
    the bag-of-words representation — demonstrating how PoS information
    can enrich a statistical NLP pipeline.

    Returns an (N, n_pos_tags) matrix of normalised PoS tag counts.
    """
    UNIVERSAL_TAGS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET",
        "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X",
    ]
    tag_to_idx = {tag: i for i, tag in enumerate(UNIVERSAL_TAGS)}
    n_tags = len(UNIVERSAL_TAGS)

    feature_matrix = np.zeros((len(texts), n_tags), dtype=np.float32)

    print("  Extracting PoS features via spaCy...")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size,
                                     disable=["ner", "parser"])):
        counts = np.zeros(n_tags, dtype=np.float32)
        for token in doc:
            idx = tag_to_idx.get(token.pos_, None)
            if idx is not None:
                counts[idx] += 1
        total = counts.sum()
        if total > 0:
            counts /= total          # normalise to relative frequency
        feature_matrix[i] = counts

    print(f"  PoS feature matrix shape: {feature_matrix.shape}")
    return feature_matrix


# ════════════════════════════════════════════════════════════════════════════
# 4. TF-IDF VECTORISER
# ════════════════════════════════════════════════════════════════════════════

def build_tfidf(train_texts: list[str],
                ngram_range: tuple = (1, 2),
                max_features: int = 20_000) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectoriser on training data.

    Concept: TF-IDF (Term Frequency–Inverse Document Frequency) converts
    raw text into a weighted sparse vector. Unigrams capture individual
    word sentiment; bigrams capture short phrases ("not good", "very bad").

    max_features=20,000 keeps the vocabulary tractable while covering
    the most informative tokens.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,          # apply log(1 + tf) to dampen high-freq terms
        strip_accents="unicode",
        min_df=2,                   # ignore tokens that appear in < 2 docs
    )
    vectorizer.fit(train_texts)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,} tokens")
    return vectorizer


# ════════════════════════════════════════════════════════════════════════════
# 5. LOGISTIC REGRESSION WITH L-BFGS
# ════════════════════════════════════════════════════════════════════════════

def build_classifier(n_classes: int) -> LogisticRegression:
    """
    Logistic Regression using L-BFGS optimiser.

    Concept: L-BFGS is a second-order quasi-Newton optimisation method.
    Unlike first-order methods (SGD, Adam) that use only gradient information,
    L-BFGS approximates the inverse Hessian to take curvature into account,
    allowing faster convergence on convex problems like logistic regression.

    For a convex loss surface (cross-entropy + L2 regularisation),
    L-BFGS typically converges in far fewer iterations than SGD.
    """
    return LogisticRegression(
        solver="lbfgs",             # second-order quasi-Newton
        max_iter=1000,
        C=1.0,                      # inverse regularisation strength
        multi_class="multinomial" if n_classes > 2 else "auto",
        class_weight="balanced",    # handles mild SST-5 class imbalance
        n_jobs=-1,
        verbose=0,
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, label_names: list[str],
             title: str = "Model", show_plot: bool = True) -> dict:
    """
    Compute accuracy, macro-F1, and per-class report.
    Optionally plots a confusion matrix.

    Concept: Accuracy alone is misleading for imbalanced classes (SST-5).
    Macro-F1 weights each class equally regardless of support, penalising
    models that ignore minority classes like 'neutral'.
    """
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred,
                                   target_names=label_names,
                                   zero_division=0)

    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro-F1 : {mf1:.4f}")
    print(f"\n{report}")

    if show_plot:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(max(5, len(label_names)*1.5),
                            max(4, len(label_names)*1.2)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion Matrix — {title}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"results/confusion_{title.replace(' ', '_')}.png", dpi=150)
        plt.show()
        print(f"  Confusion matrix saved.")

    return {"accuracy": acc, "macro_f1": mf1}


# ════════════════════════════════════════════════════════════════════════════
# 7. FULL EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(version: int = 2,
                   use_pos_features: bool = True,
                   ngram_range: tuple = (1, 2),
                   show_plot: bool = True) -> dict:
    """
    End-to-end experiment for one SST version.

    Pipeline:
      1. Load data
      2. Tokenise
      3. Fit TF-IDF on training set
      4. (Optional) Extract PoS features
      5. Combine features
      6. Train Logistic Regression with L-BFGS
      7. Evaluate on validation set
    """
    print(f"\n{'═'*55}")
    print(f"  EXPERIMENT: SST-{version}  |  "
          f"ngrams={ngram_range}  |  PoS={use_pos_features}")
    print(f"{'═'*55}")

    # ── Step 1: Load ────────────────────────────────────────────────────────
    train_texts, train_labels, val_texts, val_labels, label_names = load_sst(version)
    n_classes = len(label_names)

    # ── Step 1b: Clean ──────────────────────────────────────────────────────
    print("\n[0/4] Cleaning text (PTB brackets, contractions, whitespace)...")
    train_texts = [clean_text(t) for t in train_texts]
    val_texts   = [clean_text(t) for t in val_texts]

    # ── Step 2-3: TF-IDF ────────────────────────────────────────────────────
    print("\n[1/4] Building TF-IDF features...")
    vectorizer = build_tfidf(train_texts, ngram_range=ngram_range)
    X_train_tfidf = vectorizer.transform(train_texts)
    X_val_tfidf   = vectorizer.transform(val_texts)

    # ── Step 4: PoS features ─────────────────────────────────────────────────
    if use_pos_features:
        print("\n[2/4] Extracting PoS features...")
        from scipy.sparse import hstack, csr_matrix
        pos_train = extract_pos_features(list(train_texts))
        pos_val   = extract_pos_features(list(val_texts))
        X_train = hstack([X_train_tfidf, csr_matrix(pos_train)])
        X_val   = hstack([X_val_tfidf,   csr_matrix(pos_val)])
    else:
        X_train = X_train_tfidf
        X_val   = X_val_tfidf

    print(f"  Feature matrix shape — train: {X_train.shape}, val: {X_val.shape}")

    # ── Step 5: Train ────────────────────────────────────────────────────────
    print("\n[3/4] Training Logistic Regression (L-BFGS)...")
    clf = build_classifier(n_classes)
    t0 = time.time()
    clf.fit(X_train, train_labels)
    train_time = time.time() - t0
    print(f"  Training time : {train_time:.1f}s")
    print(f"  Iterations    : {clf.n_iter_}")

    # ── Step 6: Evaluate ─────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    title  = f"N-gram Baseline SST-{version}"
    results = evaluate(val_labels, y_pred, label_names,
                       title=title, show_plot=show_plot)
    results["train_time_s"] = round(train_time, 2)

    return results, vectorizer, clf


# ════════════════════════════════════════════════════════════════════════════
# 8. ABLATION: NGRAM RANGE COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def ngram_ablation(version: int = 2) -> pd.DataFrame:
    """
    Compare unigram-only vs. unigram+bigram vs. unigram+bigram+trigram
    to demonstrate the effect of N-gram range on classification performance.

    Concept: N-gram models capture local word sequences. Bigrams like
    "not good" or "very bad" convey sentiment that unigrams miss entirely.
    Trigrams add further context but risk sparsity.
    """
    train_texts, train_labels, val_texts, val_labels, label_names = load_sst(version)
    n_classes = len(label_names)
    rows = []

    for ngram_range in [(1, 1), (1, 2), (1, 3)]:
        label = f"{'uni' if ngram_range[1]==1 else 'uni+bi' if ngram_range[1]==2 else 'uni+bi+tri'}"
        vec   = build_tfidf(train_texts, ngram_range=ngram_range)
        clf   = build_classifier(n_classes)
        clf.fit(vec.transform(train_texts), train_labels)
        y_pred = clf.predict(vec.transform(val_texts))
        rows.append({
            "ngram_range" : str(ngram_range),
            "label"       : label,
            "accuracy"    : round(accuracy_score(val_labels, y_pred), 4),
            "macro_f1"    : round(f1_score(val_labels, y_pred, average="macro"), 4),
        })
        print(f"  {label:>15}  acc={rows[-1]['accuracy']:.4f}  f1={rows[-1]['macro_f1']:.4f}")

    df = pd.DataFrame(rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ["accuracy", "macro_f1"]):
        ax.bar(df["label"], df[metric], color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title(f"N-gram Ablation — {metric} (SST-{version})")
        ax.set_ylabel(metric)
        ax.set_ylim(df[metric].min() - 0.02, df[metric].max() + 0.02)
        for i, v in enumerate(df[metric]):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"results/ngram_ablation_sst{version}.png", dpi=150)
    plt.show()

    return df


# ════════════════════════════════════════════════════════════════════════════
# 9. TOP FEATURES ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def top_features(vectorizer: TfidfVectorizer,
                 clf: LogisticRegression,
                 label_names: list[str],
                 top_n: int = 15):
    """
    Show the most positively- and negatively-weighted n-gram features
    for each class. Provides interpretability for the bag-of-words model.
    """
    feature_names = vectorizer.get_feature_names_out()

    for class_idx, class_name in enumerate(label_names):
        if clf.coef_.shape[0] == 1:
            # Binary: single coefficient vector
            coef = clf.coef_[0] if class_idx == 1 else -clf.coef_[0]
        else:
            coef = clf.coef_[class_idx]

        top_pos = np.argsort(coef)[-top_n:][::-1]
        top_neg = np.argsort(coef)[:top_n]

        print(f"\n── Class: {class_name} ──────────────────────────────")
        print(f"  Top positive features: "
              f"{', '.join(feature_names[top_pos])}")
        print(f"  Top negative features: "
              f"{', '.join(feature_names[top_neg])}")


# ════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    all_results = {}

    # ── SST-2: binary classification ────────────────────────────────────────
    print("\n" + "▓"*55)
    print("  SST-2: BINARY SENTIMENT CLASSIFICATION")
    print("▓"*55)
    results_sst2, vec2, clf2 = run_experiment(version=2, use_pos_features=True)
    all_results["SST-2"] = results_sst2

    print("\nTop discriminative features (SST-2):")
    top_features(vec2, clf2, label_names=["negative", "positive"])

    print("\nN-gram ablation (SST-2):")
    ablation_sst2 = ngram_ablation(version=2)

    # ── SST-5: fine-grained classification ──────────────────────────────────
    print("\n" + "▓"*55)
    print("  SST-5: FINE-GRAINED SENTIMENT CLASSIFICATION")
    print("▓"*55)
    results_sst5, vec5, clf5 = run_experiment(version=5, use_pos_features=True)
    all_results["SST-5"] = results_sst5

    print("\nTop discriminative features (SST-5):")
    top_features(vec5, clf5,
                 label_names=["very negative", "negative", "neutral",
                               "positive", "very positive"])

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  BASELINE SUMMARY")
    print("═"*55)
    summary = pd.DataFrame(all_results).T
    summary.index.name = "Dataset"
    print(summary.to_string())

    # Save results
    summary.to_csv("results/baseline_results.csv")
    print("\nResults saved to results/baseline_results.csv")

    # ── Degradation: SST-2 → SST-5 ──────────────────────────────────────────
    acc_drop = results_sst2["accuracy"] - results_sst5["accuracy"]
    f1_drop  = results_sst2["macro_f1"] - results_sst5["macro_f1"]
    print(f"\n── SST-2 → SST-5 Degradation (Baseline) ─────────────")
    print(f"  Accuracy drop : -{acc_drop:.4f} ({acc_drop*100:.2f} pp)")
    print(f"  Macro-F1 drop : -{f1_drop:.4f}  ({f1_drop*100:.2f} pp)")
    print("  (This is the gap that BiRNN and BERT will attempt to close.)\n")
