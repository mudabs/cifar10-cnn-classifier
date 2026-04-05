"""
visualize.py
------------
Feature map visualisations for the CIFAR-10 CNN.

For each of three representative test images the script:
    1. Passes the image through the model.
    2. Extracts intermediate feature maps from ConvBlock1, ConvBlock2,
       and ConvBlock3 (returned directly by CIFAR10CNN.forward).
    3. Renders a grid showing the original image alongside up to
       MAX_CHANNELS (16) feature maps per block.

Selected images
---------------
    correct_easy   — correctly classified with highest model confidence
    correct_hard   — correctly classified with lowest  model confidence
    misclassified  — first image whose prediction differs from true label

Output
------
    PNG files written to output_dir/  (default: 'visualizations/').
"""

import os
import torch
import matplotlib
matplotlib.use('Agg')           # non-interactive backend; safe on all systems
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from dataset import CIFAR10FolderDataset, get_test_transform
from model   import CIFAR10CNN


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

MAX_CHANNELS = 16   # maximum channels shown per block
GRID_COLS    = 8    # channels arranged per row in the feature grid

# CIFAR-10 normalisation statistics (must match dataset.py)
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def _unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse per-channel normalisation so that images can be displayed."""
    return (tensor * _STD + _MEAN).clamp(0.0, 1.0)


# -----------------------------------------------------------------------
# Image selection
# -----------------------------------------------------------------------

def _select_images(model, dataset, device):
    """
    Scan the full test dataset and identify three representative samples.

    Strategy
    --------
        correct_easy  : highest softmax confidence among correct predictions
        correct_hard  : lowest  softmax confidence among correct predictions
        misclassified : first example where prediction ≠ true label

    Returns
    -------
        selections  : list of (title: str, info: dict | None)
                      info keys → 'image' (Tensor), 'label' (int),
                                  'pred'  (int),    'conf'  (float)
        class_names : list of class name strings
    """
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()

    correct_easy, correct_hard, misclassified = None, None, None
    best_high_conf =  -1.0
    best_low_conf  =   2.0   # above max possible softmax value (1.0)

    with torch.no_grad():
        for images, labels in loader:
            logits, _  = model(images.to(device))
            probs      = torch.softmax(logits, dim=1).cpu()
            preds      = probs.argmax(dim=1)
            confs      = probs.max(dim=1).values

            for i in range(len(labels)):
                entry = {
                    'image': images[i],         # (3, H, W) normalised tensor
                    'label': labels[i].item(),
                    'pred' : preds[i].item(),
                    'conf' : confs[i].item(),
                }

                if entry['pred'] == entry['label']:
                    # Update highest-confidence correct example
                    if entry['conf'] > best_high_conf:
                        best_high_conf = entry['conf']
                        correct_easy   = entry
                    # Update lowest-confidence correct example (hard case)
                    if entry['conf'] < best_low_conf:
                        best_low_conf = entry['conf']
                        correct_hard  = entry
                elif misclassified is None:
                    misclassified = entry  # take the first wrong prediction

    selections = [
        ('Correctly Classified (High Confidence)', correct_easy),
        ('Hard Correctly Classified (Low Confidence)', correct_hard),
        ('Misclassified',                               misclassified),
    ]
    return selections, dataset.classes


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def _plot_single_image(
    image_tensor,
    feature_maps,
    title_str:   str,
    class_names,
    label:       int,
    pred:        int,
    conf:        float,
    save_path:   str,
) -> None:
    """
    Render a multi-panel figure and save it as a PNG.

    Layout
    ------
        Row 0           : Original 32×32 input image (centred)
        Rows 1–2        : Feature maps from ConvBlock1 (up to 16 channels)
        Rows 3–4        : Feature maps from ConvBlock2
        Rows 5–6        : Feature maps from ConvBlock3

    Each channel is shown as a false-colour (viridis) heat map with no axes.
    Block labels are annotated above the first channel of each section.

    Args:
        image_tensor : (3, H, W) *normalised* image tensor.
        feature_maps : list of 3 tensors, each (C, H, W), no batch dim.
        title_str    : Descriptive title for the figure.
        class_names  : List of class name strings.
        label        : Ground-truth class index.
        pred         : Predicted class index.
        conf         : Model confidence (max softmax probability).
        save_path    : File path for the saved PNG.
    """
    rows_per_block = (MAX_CHANNELS + GRID_COLS - 1) // GRID_COLS   # = 2
    total_rows     = 1 + len(feature_maps) * rows_per_block         # = 7

    fig = plt.figure(figsize=(GRID_COLS * 2, total_rows * 2 + 1))
    gs  = gridspec.GridSpec(
        total_rows, GRID_COLS, figure=fig, hspace=0.55, wspace=0.08,
    )

    # ── Figure super-title ──────────────────────────────────────────────
    fig.suptitle(
        f"{title_str}\n"
        f"True: {class_names[label]}    "
        f"Pred: {class_names[pred]}    "
        f"Confidence: {conf:.1%}",
        fontsize=11, y=1.01,
    )

    # ── Original image — centred in the top row ─────────────────────────
    mid = GRID_COLS // 2                         # = 4
    ax0 = fig.add_subplot(gs[0, mid - 1: mid + 1])
    img = _unnormalize(image_tensor).permute(1, 2, 0).numpy()
    ax0.imshow(img, interpolation='nearest')
    ax0.set_title('Input Image', fontsize=9)
    ax0.axis('off')

    # ── Feature maps for each ConvBlock ─────────────────────────────────
    for blk_idx, fmap in enumerate(feature_maps):
        n_ch      = min(fmap.shape[0], MAX_CHANNELS)
        row_start = 1 + blk_idx * rows_per_block

        # Label placed above the block's first subplot
        block_label = (
            f"Block {blk_idx + 1}  "
            f"({fmap.shape[0]} channels, "
            f"{fmap.shape[1]}×{fmap.shape[2]} spatial)"
        )

        for ch in range(n_ch):
            r  = row_start + ch // GRID_COLS   # row within the grid
            c  = ch % GRID_COLS                # column within the grid
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(
                fmap[ch].cpu().numpy(),
                cmap='viridis',
                interpolation='nearest',
            )
            ax.axis('off')

            # Annotate block name once, above the first channel
            if ch == 0:
                ax.set_title(block_label, fontsize=7, loc='left', pad=3)

    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[Visualize] Saved: {save_path}")


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def visualize(
    data_dir:   str = 'data',
    model_path: str = 'model.pth',
    output_dir: str = 'visualizations',
) -> None:
    """
    Generate and save feature map visualisations for three test images.

    Args:
        data_dir   : Root directory containing a 'test/' sub-folder.
        model_path : Path to saved model weights.
        output_dir : Directory where output PNGs are written.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Visualize] Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    test_dir     = os.path.join(data_dir, 'test')
    test_dataset = CIFAR10FolderDataset(test_dir, transform=get_test_transform())

    # ── Model ────────────────────────────────────────────────────────────
    model = CIFAR10CNN(num_classes=len(test_dataset.classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"[Visualize] Loaded weights from '{model_path}'")
    print(f"[Visualize] Scanning {len(test_dataset)} test images …")

    # ── Select three representative images ───────────────────────────────
    selections, class_names = _select_images(model, test_dataset, device)

    # ── Plot feature maps for each selected image ─────────────────────────
    for title, info in selections:
        if info is None:
            print(f"[Visualize] WARNING — no candidate found for: '{title}'")
            continue

        # Run a single forward pass to retrieve feature maps
        img_batch = info['image'].unsqueeze(0).to(device)   # add batch dim
        with torch.no_grad():
            _, feature_maps = model(img_batch)

        # Remove the batch dimension: (1, C, H, W) → (C, H, W)
        fmaps = [fm.squeeze(0) for fm in feature_maps]

        # Build a filesystem-safe filename from the title
        safe_name = title.lower().split('(')[0].strip().replace(' ', '_')
        save_path = os.path.join(output_dir, f"{safe_name}.png")

        _plot_single_image(
            image_tensor=info['image'],
            feature_maps=fmaps,
            title_str=title,
            class_names=class_names,
            label=info['label'],
            pred=info['pred'],
            conf=info['conf'],
            save_path=save_path,
        )

    print(f"\n[Visualize] All plots written to '{output_dir}/'")
