import torch
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp

# --- MODEL 1: THE CLASSIFIER (View Detection) ---
class SpineViewClassifier(nn.Module):
    def __init__(self):
        super(SpineViewClassifier, self).__init__()
        # Use EfficientNet-B0 - Better accuracy/efficiency trade-off than ResNet
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Modify the last layer for 2 classes: [AP_View, Lateral_View]
        # EfficientNet's classifier is a Sequential block, last layer is[1]
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.backbone(x)

# --- MODEL 2: THE SEGMENTOR (U-Net++ for Regions) ---
class UNetSegmentor(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetSegmentor, self).__init__()
        # Use U-Net++ with EfficientNet-B3 backbone
        # Pretrained on ImageNet for better feature extraction
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=n_classes,
            activation=None # We apply sigmoid/softmax later or in loss
        )

    def forward(self, x):
        return self.model(x)