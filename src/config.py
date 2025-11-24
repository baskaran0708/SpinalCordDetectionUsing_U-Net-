import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image settings
IMG_SIZE = 512  # Resize all inputs to 512x512
CHANNELS = 3

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

# Class Mappings for Segmentation
# Channel 0: Cervical, Channel 1: Thoracic, Channel 2: Lumbar
NUM_SEG_CLASSES = 3 

# Paths
DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"