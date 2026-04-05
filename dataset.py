"""
dataset.py
----------
Custom PyTorch Dataset that loads CIFAR-10 images from a folder tree.

Expected layout
---------------
root/
├── airplane/
│   ├── img_00000.png
│   └── ...
├── automobile/
│   └── ...
└── ...         (10 class folders total)

Class labels are assigned alphabetically from the subfolder names, making
the mapping deterministic regardless of filesystem ordering.
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CIFAR10FolderDataset(Dataset):
    """
    Reads images from a class-per-folder directory structure and returns
    (image_tensor, label_index) pairs.

    Args:
        root_dir  (str)      : Path to the split root (e.g. 'data/train').
        transform (callable) : Optional torchvision transform applied to
                               each PIL image before returning a tensor.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        # Populated by _build_index()
        self.samples      = []   # list of (abs_path, label_index)
        self.classes      = []   # list of class name strings
        self.class_to_idx = {}   # class name → integer label

        self._build_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Scan root_dir and collect (path, label) pairs."""
        # Sort so the idx ↔ class mapping is always deterministic
        class_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.classes      = class_dirs
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_dirs)}

        for cls in class_dirs:
            cls_path = os.path.join(self.root_dir, cls)
            label    = self.class_to_idx[cls]
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_path, fname), label))

    # ------------------------------------------------------------------
    # Required Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Return:
            image_tensor (Tensor) : shape (3, H, W), dtype float32
            label        (int)    : class index
        """
        img_path, label = self.samples[idx]

        # Open as RGB so we always get 3-channel tensors (no RGBA / greyscale)
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -----------------------------------------------------------------------
# Transform factories
# -----------------------------------------------------------------------

# CIFAR-10 channel-wise mean and standard deviation (pre-computed)
_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.2023, 0.1994, 0.2010]


def get_train_transform():
    """
    Data-augmented transform for the training split.
    Augmentations help regularise the model and reduce overfitting.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),       # ensure consistent spatial size
        transforms.RandomHorizontalFlip(), # horizontal symmetry augmentation
        transforms.RandomCrop(32, padding=4),  # translation augmentation
        transforms.ToTensor(),             # HWC uint8 → CHW float32 in [0,1]
        transforms.Normalize(mean=_MEAN, std=_STD),  # zero-centre per channel
    ])


def get_test_transform():
    """
    Deterministic transform for the test/validation split.
    No augmentation — ensures reproducible evaluation.
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
