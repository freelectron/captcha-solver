import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Conv(nn.Module):
    """
    Basic convolution block used throughout YOLOv8.
    This is the fundamental building block that combines:
    - 2D Convolution
    - Batch Normalization (for training stability)
    - SiLU activation (Swish activation, performs better than ReLU in many cases)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        # Auto-calculate padding to maintain spatial dimensions when stride=1
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # Batch norm for stable training
        self.act = nn.SiLU()  # SiLU (Swish) activation: x * sigmoid(x)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """
    Bottleneck block used in C2f modules.
    This implements a residual connection pattern:
    - 1x1 conv to reduce channels (bottleneck)
    - 3x3 conv to process features
    - Skip connection if input/output channels match (residual learning)

    This design helps with gradient flow and allows deeper networks.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        # Expansion factor controls the bottleneck width (0.5 = half the output channels)
        hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, hidden_channels, 1)  # 1x1 conv (pointwise)
        self.conv2 = Conv(hidden_channels, out_channels, 3)  # 3x3 conv (depthwise)

        # Only add skip connection if dimensions match and shortcut is enabled
        self.add_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        # Apply convolutions
        out = self.conv2(self.conv1(x))

        # Add residual connection if applicable
        if self.add_shortcut:
            return x + out
        else:
            return out

class C2f(nn.Module):
    """
    C2f (Cross Stage Partial Bottleneck with 2 convolutions) module.
    This is YOLOv8's main building block, inspired by YOLOv5's C3 module.

    Key idea:
    - Split input into two paths
    - Process one path through bottleneck blocks
    - Concatenate all intermediate features
    - Final 1x1 conv to reduce channels

    This design improves gradient flow and feature reuse.
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, expansion=0.5):
        super().__init__()
        # Calculate intermediate channel count
        self.c = int(out_channels * expansion)  # Hidden channels

        # First conv splits input and expands to 2 * hidden_channels
        self.cv1 = Conv(in_channels, 2 * self.c, 1)

        # Final conv takes concatenated features and outputs desired channels
        # (2 + n) because we concatenate: 2 initial splits + n bottleneck outputs
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)

        # Stack of bottleneck blocks
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(self.c, self.c, shortcut, expansion=1.0) for _ in range(n)]
        )

    def forward(self, x):
        # Split input into two equal parts along channel dimension
        y = list(self.cv1(x).chunk(2, 1))  # Split into 2 parts

        # Process through bottleneck blocks, each taking output from previous
        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))  # Append output of each bottleneck

        # Concatenate all features and apply final convolution
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast (SPPF) layer.
    This module captures multi-scale features by applying max pooling at different scales.

    Instead of parallel pooling (expensive), it uses sequential pooling:
    - More efficient than original SPP
    - Maintains similar receptive field expansion
    - Helps detect objects at different scales
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2

        # Reduce channels first
        self.cv1 = Conv(in_channels, hidden_channels, 1)

        # Expand channels after pooling (4x because we concat 4 feature maps)
        self.cv2 = Conv(hidden_channels * 4, out_channels, 1)

        # Max pooling with stride=1 preserves spatial dimensions
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # Reduce channels
        x = self.cv1(x)

        # Sequential max pooling to create pyramid
        y1 = self.maxpool(x)      # 1st level
        y2 = self.maxpool(y1)     # 2nd level
        y3 = self.maxpool(y2)     # 3rd level

        # Concatenate original + 3 pooled versions = 4x channels
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class DetectHead(nn.Module):
    def __init__(self, num_classes, in_channels=(128, 256, 512), num_boxes=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_box_coords = 4
        self.num_outputs = num_classes + self.num_box_coords
        self.num_boxes = num_boxes

        ch = sum(in_channels)
        self.fc = nn.Linear(ch, self.num_outputs * self.num_boxes)

    def forward(self, features):
        # Pool each feature to 1x1 and concat along channel dimension
        x = torch.cat([F.adaptive_avg_pool2d(f, 1) for f in features], dim=1)
        x = x.flatten(1)  # shape: [B, ch]
        x = self.fc(x)
        x = x.view(x.size(0), self.num_boxes, self.num_outputs)
        return [x]


class YOLOv8Nano(nn.Module):
    """
    YOLOv8 Nano model - the smallest and fastest version.

    Architecture:
    1. Backbone: Feature extraction with multiple scales
    2. Neck: Feature fusion using FPN + PAN
    3. Head: Final predictions

    Nano configuration:
    - depth_multiple = 0.33 (fewer layers)
    - width_multiple = 0.25 (fewer channels)
    - Optimized for speed and memory efficiency
    """

    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # YOLOv8n configuration parameters
        self.depth_multiple = 0.33  # Controls number of layers
        self.width_multiple = 0.25  # Controls channel width

        # Calculate channel sizes for nano model
        # These are much smaller than larger models for efficiency
        self.ch1 = max(round(64 * self.width_multiple), 1)    # 16 channels
        self.ch2 = max(round(128 * self.width_multiple), 1)   # 32 channels
        self.ch3 = max(round(256 * self.width_multiple), 1)   # 64 channels
        self.ch4 = max(round(512 * self.width_multiple), 1)   # 128 channels
        self.ch5 = max(round(1024 * self.width_multiple), 1)  # 256 channels

        # Calculate depth (number of bottleneck blocks)
        self.n1 = max(round(3 * self.depth_multiple), 1)  # 1 block
        self.n2 = max(round(6 * self.depth_multiple), 1)  # 2 blocks

        # Build model components
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = DetectHead(num_classes, (self.ch3, self.ch4, self.ch5))

        # Initialize weights for better convergence
        self._initialize_weights()

    def _build_backbone(self):
        """
        Build the backbone network for feature extraction.

        The backbone progressively downsamples the input while increasing channel depth:
        - P1: 640x640 -> 320x320 (stride 2)
        - P2: 320x320 -> 160x160 (stride 2)
        - P3: 160x160 -> 80x80 (stride 2)   <- Output for detection
        - P4: 80x80 -> 40x40 (stride 2)     <- Output for detection
        - P5: 40x40 -> 20x20 (stride 2)     <- Output for detection
        """
        return nn.ModuleList([
            # Stage 0: Initial convolution
            Conv(3, self.ch1, 3, 2),                    # P1/2: 640->320

            # Stage 1: First downsampling
            Conv(self.ch1, self.ch2, 3, 2),             # P2/4: 320->160
            C2f(self.ch2, self.ch2, self.n1, True),     # Process P2 features

            # Stage 2: Second downsampling (P3 - used for detection)
            Conv(self.ch2, self.ch3, 3, 2),             # P3/8: 160->80
            C2f(self.ch3, self.ch3, self.n2, True),     # Process P3 features

            # Stage 3: Third downsampling (P4 - used for detection)
            Conv(self.ch3, self.ch4, 3, 2),             # P4/16: 80->40
            C2f(self.ch4, self.ch4, self.n2, True),     # Process P4 features

            # Stage 4: Fourth downsampling (P5 - used for detection)
            Conv(self.ch4, self.ch5, 3, 2),             # P5/32: 40->20
            C2f(self.ch5, self.ch5, self.n1, True),     # Process P5 features

            # Stage 5: Spatial Pyramid Pooling
            SPPF(self.ch5, self.ch5, 5),                # Multi-scale feature aggregation
        ])

    # def _build_neck(self):
    #     """
    #     Build the neck network for feature fusion.
    #
    #     Uses FPN (Feature Pyramid Network) + PAN (Path Aggregation Network):
    #     - FPN: Top-down pathway (large to small features)
    #     - PAN: Bottom-up pathway (small to large features)
    #
    #     This allows the model to combine features from different scales,
    #     helping detect objects of various sizes.
    #     """
    #     return nn.ModuleDict({
    #         # FPN - Top-down pathway
    #         'upsample1': nn.Upsample(None, 2, 'nearest'),        # 2x upsampling
    #         'c2f_up1': C2f(self.ch5 + self.ch4, self.ch4, self.n1),  # Fuse P5+P4
    #
    #         'upsample2': nn.Upsample(None, 2, 'nearest'),        # 2x upsampling
    #         'c2f_up2': C2f(self.ch4 + self.ch3, self.ch3, self.n1),  # Fuse P4+P3
    #
    #         # PAN - Bottom-up pathway
    #         'downsample1': Conv(self.ch3, self.ch3, 3, 2),       # 2x downsampling
    #         'c2f_down1': C2f(self.ch3 + self.ch4, self.ch4, self.n1),  # Fuse P3+P4
    #
    #         'downsample2': Conv(self.ch4, self.ch4, 3, 2),       # 2x downsampling
    #         'c2f_down2': C2f(self.ch4 + self.ch5, self.ch5, self.n1),  # Fuse P4+P5
    #     })

    def forward(self, x):
        """
        Forward pass through the entire network.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            List of prediction tensors for different scales
        """
        # Backbone: Extract multi-scale features
        backbone_features = []

        for i, layer in enumerate(self.backbone):
            x = layer(x)
            # Save features from P3, P4, P5 levels for detection
            if i in [4, 6, 9]:  # After C2f blocks of stages 2, 3, 4
                backbone_features.append(x)

        p3, p4, p5 = backbone_features  # that would go to YOLOv8 neck

        # Skipped the neck, doing single detect with
        self.head([p3, p4, p5])

        # # Neck: Feature fusion
        # # FPN - Top-down pathway (combine large-scale features with small-scale)
        # up1 = self.neck['upsample1'](p5)                    # Upsample P5
        # fused1 = torch.cat([up1, p4], dim=1)                # Concatenate with P4
        # n1 = self.neck['c2f_up1'](fused1)                   # Process fused features
        #
        # up2 = self.neck['upsample2'](n1)                    # Upsample fused P4
        # fused2 = torch.cat([up2, p3], dim=1)                # Concatenate with P3
        # n2 = self.neck['c2f_up2'](fused2)                   # Process fused features
        #
        # # PAN - Bottom-up pathway (refine features with high-resolution info)
        # down1 = self.neck['downsample1'](n2)                # Downsample refined P3
        # fused3 = torch.cat([down1, n1], dim=1)              # Concatenate with refined P4
        # n3 = self.neck['c2f_down1'](fused3)                 # Process fused features
        #
        # down2 = self.neck['downsample2'](n3)                # Downsample refined P4
        # fused4 = torch.cat([down2, p5], dim=1)              # Concatenate with P5
        # n4 = self.neck['c2f_down2'](fused4)                 # Process fused features

        # Head: Generate final predictions
        # n2 = P3 level (80x80, small objects)
        # n3 = P4 level (40x40, medium objects)
        # n4 = P5 level (20x20, large objects)
        # return self.head([n2, n3, n4])

    def _initialize_weights(self):
        """
        Initialize model weights for better training convergence.

        Uses Kaiming initialization for conv layers and standard initialization for BN.
        Proper initialization is crucial for deep networks to train effectively.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization accounts for ReLU-like activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                # Standard BN initialization
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

# Placeholder loss function (replace with YOLOv8-style loss for production)
class YoloSimpleLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes

    def forward(self, preds, targets):
        # preds: list of [B, 4+num_classes, H, W] for each scale
        # targets: list of ground truth per scale (implement target assignment as needed)
        loss = 0.0
        for pred, target in zip(preds, targets):
            box_pred = pred[:, :4, :, :]
            class_pred = pred[:, 4:, :, :]
            box_target = target[:, :4, :, :]
            class_target = target[:, 4:, :, :]
            loss += self.bce(box_pred, box_target)
            loss += self.bce(class_pred, class_target)

        return loss


# Example usage and testing
if __name__ == "__main__":
    # Check device availability
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')  # NVIDIA GPU
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')   # CPU fallback
        print("Using CPU")

    # Create model for CAPTCHA character detection (36 classes: 0-9, A-Z)
    num_classes = 36
    model = YOLOv8Nano(num_classes=num_classes)
    model = model.to(device)

    print(f"Created YOLOv8 Nano model with {num_classes} classes")

    # Test forward pass with dummy input
    batch_size = 1
    input_w = 160
    input_h = 60
    dummy_input = torch.randn(batch_size, 3, input_h, input_w).to(device)

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input)

        print(f"\nModel successfully processed input of shape: {dummy_input.shape}")
        print(f"Number of prediction scales: {len(predictions)}")

        for i, pred in enumerate(predictions):
            print(f"Scale {i+1} prediction shape: {pred.shape}")

    import os
    os.exit(1)

    import torch.optim as optim

    # Training loop
    def train_yolov8_nano(model, dataloader, num_epochs=10, lr=1e-3, device='cpu'):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = YoloSimpleLoss(num_classes=model.num_classes)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for images, targets in dataloader:
                images = images.to(device)
                # targets: list of tensors per scale, each [B, 4+num_classes, H, W]
                targets = [t.to(device) for t in targets]

                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")


    from torch.utils.data import Dataset, DataLoader

    class Chaptcha100kDataset(Dataset):
        def __init__(self, annot_folder, img_folder):
            pass
    train_loader = Chaptcha100kDataset(annot_folder="path/to/annotations.json")

    train_yolov8_nano(model, train_loader, num_epochs=2, lr=1e-3, device=device)

    # Calculate and display model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # Assume 4 bytes per parameter

    print("\nYOLOv8 Nano model created and tested successfully!")
    print("Ready for training on your CAPTCHA dataset!")