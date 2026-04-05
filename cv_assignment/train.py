"""
train.py
--------
Training loop for the CIFAR-10 CNN.

Hyperparameters
---------------
    batch_size    : 64
    learning_rate : 0.001
    num_epochs    : 10
    optimizer     : Adam
    loss          : CrossEntropyLoss

Tracking
--------
    Training accuracy is computed and printed after every epoch.

Output
------
    Trained weights saved to model_save_path (default: 'model.pth').
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CIFAR10FolderDataset, get_train_transform
from model   import CIFAR10CNN
from utils   import compute_accuracy


def train(
    data_dir:        str   = 'data',
    model_save_path: str   = 'model.pth',
    num_epochs:      int   = 10,
    batch_size:      int   = 64,
    lr:              float = 0.001,
    seed:            int   = 42,
) -> None:
    """
    Train the CNN on the CIFAR-10 training split.

    Args:
        data_dir        : Root directory containing a 'train/' sub-folder.
        model_save_path : File path where trained weights will be saved.
        num_epochs      : Number of complete passes through the training set.
        batch_size      : Number of images per gradient update.
        lr              : Learning rate passed to the Adam optimiser.
        seed            : Random seed for reproducible weight initialisation.
    """

    # ── Reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(seed)

    # ── Device selection ─────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device : {device}")

    # ── Dataset & DataLoader ─────────────────────────────────────────────
    train_dir     = os.path.join(data_dir, 'train')
    train_dataset = CIFAR10FolderDataset(train_dir, transform=get_train_transform())
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
    )

    print(f"[Train] Images : {len(train_dataset)}")
    print(f"[Train] Classes: {train_dataset.classes}")

    # ── Model ────────────────────────────────────────────────────────────
    model = CIFAR10CNN(num_classes=len(train_dataset.classes)).to(device)

    # ── Loss & Optimiser ─────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ────────────────────────────────────────────────────
    sep = '─' * 54
    print(f"\n{sep}")
    print(f"{'Epoch':<8}{'Loss':>12}{'Train Acc':>14}")
    print(sep)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss  = 0.0
        correct       = 0
        total         = 0

        for images, labels in train_loader:
            # Move data to the target device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass — feature maps are not needed during training
            optimizer.zero_grad()
            logits, _ = model(images)
            loss      = criterion(logits, labels)

            # Backward pass & optimiser step
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            n             = images.size(0)
            running_loss += loss.item() * n
            correct      += (logits.argmax(dim=1) == labels).sum().item()
            total        += n

        epoch_loss = running_loss / total
        epoch_acc  = correct / total * 100
        print(f"  {epoch:<6d}  {epoch_loss:>10.4f}  {epoch_acc:>12.2f}%")

    print(f"{sep}\n")

    # ── Persist model weights ─────────────────────────────────────────────
    torch.save(model.state_dict(), model_save_path)
    print(f"[Train] Weights saved → '{model_save_path}'")
