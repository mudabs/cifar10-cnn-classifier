"""
prepare_data.py
---------------
One-time utility that downloads CIFAR-10 from the official source and
organises images into the folder structure expected by dataset.py:

    data/
    ├── train/
    │   ├── airplane/      (5 000 PNG images)
    │   ├── automobile/    (5 000 PNG images)
    │   └── ...            (10 classes × 5 000 = 50 000 total)
    └── test/
        ├── airplane/      (1 000 PNG images)
        └── ...            (10 classes × 1 000 = 10 000 total)

This script does NOT use torchvision.datasets.CIFAR10 — it reads the
raw pickle files produced by the official CIFAR-10 release directly.

Usage
-----
    python prepare_data.py
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

CIFAR10_URL  = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
ARCHIVE_NAME = 'cifar-10-python.tar.gz'
TEMP_DIR     = 'cifar_temp'    # extracted raw files land here
DATA_DIR     = 'data'          # final organised images go here

# Official CIFAR-10 class order (label index 0-9)
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


# -----------------------------------------------------------------------
# Step 1 — Download
# -----------------------------------------------------------------------

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct  = min(downloaded / total_size * 100, 100)
        dl_mb = min(downloaded, total_size) / 1_048_576
        tot_mb = total_size / 1_048_576
        print(f"  {dl_mb:6.1f} / {tot_mb:.1f} MB  ({pct:.0f}%)",
              end='\r', flush=True)
    else:
        print(f"  {downloaded / 1_048_576:.1f} MB downloaded",
              end='\r', flush=True)


def _download(url: str = CIFAR10_URL, dest: str = ARCHIVE_NAME) -> None:
    if os.path.exists(dest):
        print(f"[Setup] Archive already present: {dest}")
        return
    print(f"[Setup] Downloading CIFAR-10 (~170 MB) …")
    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()   # newline after progress output
    print("[Setup] Download complete.")


# -----------------------------------------------------------------------
# Step 2 — Extract
# -----------------------------------------------------------------------

def _extract(archive: str = ARCHIVE_NAME, dest: str = TEMP_DIR) -> None:
    if os.path.exists(dest):
        print(f"[Setup] Already extracted: {dest}/")
        return
    print(f"[Setup] Extracting {archive} …")
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(dest)
    print("[Setup] Extraction complete.")


# -----------------------------------------------------------------------
# Step 3 — Save images to folder structure
# -----------------------------------------------------------------------

def _save_batch(
    batch_path: str,
    split:      str,
    offset:     int,
    data_dir:   str,
) -> None:
    """
    Read one CIFAR-10 pickle batch and write each image as a PNG file.

    CIFAR-10 binary format
    ----------------------
        data   : ndarray uint8, shape (N, 3072)
                 3072 = 3 channels × 32 × 32 pixels, stored channel-first.
        labels : list of N integer class indices (0-9).

    Args:
        batch_path : Path to the raw pickle file.
        split      : 'train' or 'test'.
        offset     : Added to the local index to create unique filenames
                     across multiple training batches.
        data_dir   : Root output directory.
    """
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3) for PIL
    images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']

    for idx, (img_arr, label) in enumerate(zip(images, labels)):
        cls_name = CLASSES[label]
        cls_dir  = os.path.join(data_dir, split, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        # Use a global index so filenames don't collide across batches
        img_path = os.path.join(cls_dir, f"img_{offset + idx:05d}.png")
        if not os.path.exists(img_path):   # skip if already written
            Image.fromarray(img_arr).save(img_path)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def prepare(data_dir: str = DATA_DIR) -> None:
    """
    Download CIFAR-10, extract, and organise images into data_dir/.

    Args:
        data_dir : Root output directory (default: 'data').
    """
    _download()
    _extract()

    batch_dir = os.path.join(TEMP_DIR, 'cifar-10-batches-py')

    # Training data: 5 batches × 10 000 images = 50 000 images
    print("[Setup] Writing training images …")
    for i in range(1, 6):
        batch_path = os.path.join(batch_dir, f'data_batch_{i}')
        _save_batch(batch_path, split='train',
                    offset=(i - 1) * 10_000, data_dir=data_dir)
        print(f"  training batch {i}/5 done")

    # Test data: 1 batch × 10 000 images
    print("[Setup] Writing test images …")
    _save_batch(os.path.join(batch_dir, 'test_batch'),
                split='test', offset=0, data_dir=data_dir)
    print("  test batch done")

    print(f"\n[Setup] Done!  CIFAR-10 images organised in '{data_dir}/'")
    print("[Setup] Run:  python main.py  →  choose option 1 to train.")


if __name__ == '__main__':
    prepare()
