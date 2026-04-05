"""
evaluate.py
-----------
Evaluation of the trained CIFAR-10 CNN on the held-out test split.

Outputs
-------
    • Overall test accuracy
    • Per-class accuracy table
    • Confusion matrix (printed as text + saved as PNG)
"""

import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import CIFAR10FolderDataset, get_test_transform
from model   import CIFAR10CNN
from utils   import compute_accuracy, per_class_accuracy, plot_confusion_matrix


def evaluate(
    data_dir:     str = 'data',
    model_path:   str = 'model.pth',
    batch_size:   int = 64,
    cm_save_path: str = 'confusion_matrix.png',
) -> None:
    """
    Run full evaluation on the test split.

    Args:
        data_dir     : Root directory containing a 'test/' sub-folder.
        model_path   : Path to saved model weights (produced by train.py).
        batch_size   : Mini-batch size for inference.
        cm_save_path : File path where the confusion matrix PNG is written.
    """

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Device: {device}")

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    test_dir     = os.path.join(data_dir, 'test')
    test_dataset = CIFAR10FolderDataset(test_dir, transform=get_test_transform())
    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,       # fixed order for reliable confusion matrix
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
    )

    class_names = test_dataset.classes
    print(f"[Evaluate] Test images : {len(test_dataset)}")
    print(f"[Evaluate] Classes     : {class_names}")

    # ── Model ────────────────────────────────────────────────────────────
    model = CIFAR10CNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Evaluate] Loaded weights from '{model_path}'")

    # ── Inference ────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            logits, _ = model(images)             # discard feature maps
            preds     = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Compute metrics ───────────────────────────────────────────────────
    overall_acc = compute_accuracy(all_labels, all_preds)
    per_cls_acc = per_class_accuracy(all_labels, all_preds, len(class_names))
    cm          = confusion_matrix(all_labels, all_preds)

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'═' * 48}")
    print(f"  Overall Test Accuracy : {overall_acc:.2f}%")
    print(f"{'═' * 48}")

    print("\nPer-Class Accuracy:")
    print(f"  {'Class':<14}  {'Correct / Total':>16}  {'Accuracy':>9}")
    print(f"  {'─' * 44}")
    for idx, cls in enumerate(class_names):
        acc = per_cls_acc[idx]
        print(f"  {cls:<14}  {'':>16}  {acc:>8.2f}%")

    # Text confusion matrix
    print("\nConfusion Matrix  (rows = true label, cols = predicted label):")
    header = " " * 12 + "  ".join(f"{c[:5]:>5}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:10]:<10}  " + "  ".join(f"{v:>5}" for v in row)
        print(row_str)

    # ── Save confusion matrix PNG ─────────────────────────────────────────
    plot_confusion_matrix(cm, class_names, save_path=cm_save_path)
    print(f"\n[Evaluate] Confusion matrix saved → '{cm_save_path}'")
