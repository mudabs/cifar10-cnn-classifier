"""
utils.py
--------
Shared helper functions used across train.py, evaluate.py, and visualize.py.

Functions
---------
    compute_accuracy      — overall accuracy as a percentage
    per_class_accuracy    — per-class accuracy as a list of percentages
    plot_confusion_matrix — render and save a confusion matrix heatmap
"""

from typing import List

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend; safe on all platforms
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Accuracy helpers
# -----------------------------------------------------------------------

def compute_accuracy(labels: List[int], preds: List[int]) -> float:
    """
    Compute overall classification accuracy.

    Args:
        labels : Ground-truth class indices.
        preds  : Predicted class indices (same length as labels).

    Returns:
        Accuracy in the range [0.0, 100.0].
    """
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    return float((labels == preds).mean() * 100)


def per_class_accuracy(
    labels:      List[int],
    preds:       List[int],
    num_classes: int,
) -> List[float]:
    """
    Compute per-class accuracy (macro, per class).

    Args:
        labels      : Ground-truth class indices.
        preds       : Predicted class indices.
        num_classes : Total number of classes.

    Returns:
        List of length num_classes; each entry is an accuracy percentage
        for that class.  Returns 0.0 for classes not present in labels.
    """
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    accs   = []
    for cls in range(num_classes):
        mask = labels == cls       # boolean mask for samples of this class
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            accs.append(float((preds[mask] == cls).mean() * 100))
    return accs


# -----------------------------------------------------------------------
# Confusion matrix visualisation
# -----------------------------------------------------------------------

def plot_confusion_matrix(
    cm:          np.ndarray,
    class_names: List[str],
    save_path:   str = 'confusion_matrix.png',
) -> None:
    """
    Render a colour-coded confusion matrix and save it to disk.

    Each cell is annotated with its count.  Cells above half the maximum
    value use white text for readability on a dark background.

    Args:
        cm          : Square array of shape (num_classes, num_classes).
        class_names : Ordered list of class name strings (matches cm axes).
        save_path   : Destination file path for the PNG.
    """
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(n + 2, n + 1))

    # Colour map — darker blue == more samples predicted in that cell
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label',      fontsize=10)
    ax.set_title('Confusion Matrix', fontsize=12)

    # Annotate each cell with its integer count
    threshold = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > threshold else 'black',
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
