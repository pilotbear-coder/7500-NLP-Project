"""
metrics.py
==========
Shared evaluation functions used across all three model tiers.
Import this in any notebook or model script.

Usage:
    from evaluation.metrics import evaluate, plot_learning_curve, results_table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)


def evaluate(y_true, y_pred, label_names: list,
             title: str = "Model", save_path: str = None,
             show_plot: bool = True) -> dict:
    """
    Compute and display accuracy, macro-F1, per-class report,
    and confusion matrix.

    Parameters
    ----------
    y_true      : ground truth labels
    y_pred      : model predictions
    label_names : list of class name strings
    title       : display name for this model/experiment
    save_path   : if provided, saves confusion matrix PNG here
    show_plot   : whether to display the confusion matrix plot

    Returns
    -------
    dict with keys: accuracy, macro_f1
    """
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
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
        fig_size = max(5, len(label_names) * 1.4)
        plt.figure(figsize=(fig_size, fig_size * 0.85))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title(f"Confusion Matrix — {title}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved: {save_path}")
        plt.show()

    return {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}


def plot_learning_curve(train_losses: list, val_losses: list,
                        train_accs: list, val_accs: list,
                        title: str = "Model", save_path: str = None):
    """
    Plot training and validation loss + accuracy curves over epochs.
    Used for BiRNN (Tier 2) and BERT (Tier 3).

    Parameters
    ----------
    train_losses : list of per-epoch training losses
    val_losses   : list of per-epoch validation losses
    train_accs   : list of per-epoch training accuracies
    val_accs     : list of per-epoch validation accuracies
    title        : display name
    save_path    : optional PNG save path
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, "o-", label="Train", color="#4C72B0")
    ax1.plot(epochs, val_losses,   "s--", label="Val",   color="#DD8452")
    ax1.set_title(f"Loss — {title}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, "o-", label="Train", color="#4C72B0")
    ax2.plot(epochs, val_accs,   "s--", label="Val",   color="#DD8452")
    ax2.set_title(f"Accuracy — {title}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.show()


def results_table(results_dict: dict) -> pd.DataFrame:
    """
    Build a summary comparison table from a dict of model results.

    Parameters
    ----------
    results_dict : {model_name: {dataset: {accuracy, macro_f1}}}

    Example
    -------
    results = {
        'N-gram Baseline': {'SST-2': {'accuracy': 0.84, 'macro_f1': 0.83},
                            'SST-5': {'accuracy': 0.43, 'macro_f1': 0.39}},
        'BiRNN + GloVe':   {'SST-2': {'accuracy': 0.89, 'macro_f1': 0.88},
                            'SST-5': {'accuracy': 0.48, 'macro_f1': 0.44}},
    }
    """
    rows = []
    for model, datasets in results_dict.items():
        for dataset, metrics in datasets.items():
            rows.append({
                "Model":     model,
                "Dataset":   dataset,
                "Accuracy":  metrics.get("accuracy", None),
                "Macro-F1":  metrics.get("macro_f1", None),
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Model", columns="Dataset",
                           values=["Accuracy", "Macro-F1"])
    return pivot.round(4)


def degradation_report(results_dict: dict):
    """
    For each model in results_dict, print the SST-2 → SST-5 accuracy
    and F1 degradation. Highlights the core analytical finding.
    """
    print(f"\n{'═'*60}")
    print(f"  SST-2 → SST-5 DEGRADATION REPORT")
    print(f"{'═'*60}")
    print(f"  {'Model':<30} {'Acc Drop':>10} {'F1 Drop':>10}")
    print(f"  {'─'*50}")
    for model, datasets in results_dict.items():
        if "SST-2" in datasets and "SST-5" in datasets:
            acc_drop = datasets["SST-2"]["accuracy"] - datasets["SST-5"]["accuracy"]
            f1_drop  = datasets["SST-2"]["macro_f1"] - datasets["SST-5"]["macro_f1"]
            print(f"  {model:<30} {acc_drop:>+9.4f}  {f1_drop:>+9.4f}")
    print(f"{'═'*60}")
