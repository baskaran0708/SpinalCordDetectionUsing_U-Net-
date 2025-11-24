import torch
import torch.nn as nn
from src.models import SpineViewClassifier, UNetSegmentor
from src.config import *

def export_models():
    print("Loading PyTorch models...")
    
    # 1. Load Classifier
    classifier = SpineViewClassifier()
    # classifier.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/view_classifier.pth"))
    classifier.eval()
    
    # 2. Load Segmentor
    segmentor = UNetSegmentor(n_classes=3)
    segmentor.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/region_segmentor.pth"))
    segmentor.eval()

    # 3. Create Dummy Input (Batch Size 1, 3 Channels, 512x512)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # 4. Export Classifier
    print("Exporting Classifier to ONNX...")
    torch.onnx.export(classifier, 
                      dummy_input, 
                      "checkpoints/view_classifier.onnx", 
                      input_names=['input'], 
                      output_names=['output'],
                      opset_version=11)

    # 5. Export Segmentor
    print("Exporting Segmentor to ONNX...")
    torch.onnx.export(segmentor, 
                      dummy_input, 
                      "checkpoints/region_segmentor.onnx", 
                      input_names=['input'], 
                      output_names=['output'],
                      opset_version=11)
                      
    print("Success! ONNX files created in /checkpoints")

if __name__ == "__main__":
    export_models()