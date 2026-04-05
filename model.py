"""
model.py
--------
CNN architecture for CIFAR-10 classification.

Architecture
------------
Stem → ConvBlock1 → MaxPool → ConvBlock2 → Downsample
     → ConvBlock3 → AdaptiveAvgPool → Flatten → MLP → Output

Channel progression: 3 (RGB) → 32 → 64 → 128

Each ConvBlock contains (all followed by BatchNorm + ReLU):
    1. Pointwise Conv  (1×1)          — cheap channel projection
    2. Depthwise Conv  (3×3, groups)  — per-channel spatial filtering
    3. Standard Conv   (3×3)          — first feature refinement
    4. Standard Conv   (3×3)          — second feature refinement

The forward() method returns BOTH class scores AND intermediate feature maps:
    forward(x) → (logits, [feat1, feat2, feat3])

Feature maps are saved after each ConvBlock (before pooling/downsampling)
and are used by visualize.py to inspect what each block has learned.
"""

import torch
import torch.nn as nn


# -----------------------------------------------------------------------
# ConvBlock — the core building block
# -----------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Composite convolutional block that mixes depthwise-separable convolutions
    with two standard 3×3 convolutions for richer spatial feature extraction.

    Sequence of operations (each followed by BatchNorm + ReLU):
        pw_conv  : 1×1 conv  — project in_channels → out_channels
        dw_conv  : 3×3 conv  — depthwise, groups=out_channels
        conv1    : 3×3 conv  — standard convolution #1
        conv2    : 3×3 conv  — standard convolution #2

    All convolutions use padding=1 where needed to preserve spatial size.

    Args:
        in_channels  (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # 1. Pointwise (1×1) projection — mixes channels without spatial ops
        self.pw_conv = nn.Conv2d(in_channels,  out_channels,
                                 kernel_size=1, bias=False)
        self.pw_bn   = nn.BatchNorm2d(out_channels)

        # 2. Depthwise (3×3) convolution — one filter per channel
        #    groups=out_channels means each channel is filtered independently
        self.dw_conv = nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, padding=1,
                                 groups=out_channels, bias=False)
        self.dw_bn   = nn.BatchNorm2d(out_channels)

        # 3. First standard 3×3 convolution — cross-channel + spatial
        self.conv1 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 4. Second standard 3×3 convolution — further feature refinement
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                                kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pointwise projection
        x = self.relu(self.pw_bn(self.pw_conv(x)))
        # Depthwise spatial filtering
        x = self.relu(self.dw_bn(self.dw_conv(x)))
        # Standard convolution refinement
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# -----------------------------------------------------------------------
# Full CNN model
# -----------------------------------------------------------------------

class CIFAR10CNN(nn.Module):
    """
    Full CNN classifier for CIFAR-10.

    Returns (logits, feature_maps) on every forward pass so callers can
    access intermediate activations without hooks:

        logits       : Tensor of shape (B, num_classes)
        feature_maps : [feat1, feat2, feat3]
                         feat1 → after ConvBlock1  shape (B, 32,  H1, W1)
                         feat2 → after ConvBlock2  shape (B, 64,  H2, W2)
                         feat3 → after ConvBlock3  shape (B, 128, H3, W3)

    For 32×32 CIFAR-10 inputs the spatial sizes are:
        feat1 : (B, 32,  32, 32)
        feat2 : (B, 64,  16, 16)
        feat3 : (B, 128,  8,  8)

    Args:
        num_classes (int): Number of output classes. Default: 10.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # ── Stem ────────────────────────────────────────────────────────
        # Lift 3 RGB channels to 32 feature channels at full resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── ConvBlock 1 — 32 → 32 channels, 32×32 ───────────────────────
        self.block1 = ConvBlock(32, 32)

        # ── MaxPool — 32×32 → 16×16 ─────────────────────────────────────
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── ConvBlock 2 — 32 → 64 channels, 16×16 ───────────────────────
        self.block2 = ConvBlock(32, 64)

        # ── Downsample — 16×16 → 8×8 (learned strided convolution) ──────
        # A strided convolution is used instead of pooling so the network
        # can learn *how* to subsample spatial information.
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── ConvBlock 3 — 64 → 128 channels, 8×8 ───────────────────────
        self.block3 = ConvBlock(64, 128)

        # ── AdaptiveAvgPool — 8×8 → 4×4 ────────────────────────────────
        # Retains spatial structure (vs. global pooling) for a richer MLP
        # input while remaining size-agnostic at inference time.
        self.avg_pool = nn.AdaptiveAvgPool2d(4)   # → (B, 128, 4, 4)

        # ── MLP classifier ───────────────────────────────────────────────
        # 128 × 4 × 4 = 2048 features → hidden layer of 512 → 10 classes
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),          # regularisation
            nn.Linear(512, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): Input batch of shape (B, 3, 32, 32).

        Returns:
            logits       (Tensor): Shape (B, num_classes) — raw scores.
            feature_maps (list)  : [feat1, feat2, feat3] as described above.
        """
        # ── Stem: (B, 3, 32, 32) → (B, 32, 32, 32) ─────────────────────
        x = self.stem(x)

        # ── Block 1 → save feature map 1 ────────────────────────────────
        feat1 = self.block1(x)          # (B, 32, 32, 32)
        x     = self.maxpool(feat1)     # (B, 32, 16, 16)

        # ── Block 2 → save feature map 2 ────────────────────────────────
        feat2 = self.block2(x)          # (B, 64, 16, 16)
        x     = self.downsample(feat2)  # (B, 64,  8,  8)

        # ── Block 3 → save feature map 3 ────────────────────────────────
        feat3 = self.block3(x)          # (B, 128, 8, 8)

        # ── Classifier head ─────────────────────────────────────────────
        x      = self.avg_pool(feat3)   # (B, 128, 4, 4)
        x      = torch.flatten(x, 1)   # (B, 2048)
        logits = self.classifier(x)     # (B, num_classes)

        return logits, [feat1, feat2, feat3]
