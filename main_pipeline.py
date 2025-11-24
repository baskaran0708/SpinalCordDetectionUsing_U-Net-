import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.models import SpineViewClassifier, UNetSegmentor
from src.config import *
from src.utils import get_region_bounds, determine_spine_shape
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_models():
    # Load View Classifier
    classifier = SpineViewClassifier().to(DEVICE)
    # classifier.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/view_classifier.pth"))
    classifier.eval()
    
    # Load Segmentor
    segmentor = UNetSegmentor(n_classes=3).to(DEVICE)
    segmentor.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/region_segmentor.pth"))
    segmentor.eval()
    
    return classifier, segmentor

def analyze_patient(image_path):
    print(f"Analyzing: {image_path}")
    clf, seg = load_models()
    
    # 1. Preprocess Image
    original_img = cv2.imread(image_path)
    original_h, original_w = original_img.shape[:2]
    
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(),
        ToTensorV2()
    ])
    
    img_tensor = transform(image=original_img)["image"].unsqueeze(0).to(DEVICE)
    
    # 2. Step 1: Classify View (AP vs Lateral)
    with torch.no_grad():
        view_logits = clf(img_tensor)
        view_pred = torch.argmax(view_logits, dim=1).item()
        view_label = "Lateral" if view_pred == 1 else "AP_View"
    
    print(f"Detected View: {view_label}")
    
    # 3. Step 2: Segment Regions
    with torch.no_grad():
        mask_pred = seg(img_tensor) # Output: [1, 3, 512, 512]
        
    # Resize mask back to original image size for calculations
    mask_np = mask_pred.squeeze().cpu().numpy()
    mask_resized = []
    for i in range(3):
        m = cv2.resize(mask_np[i], (original_w, original_h))
        mask_resized.append(m)
    
    mask_resized = np.array(mask_resized) # [3, H, W]
    
    # 4. Step 3: Extract Logic
    # Channel 0: Cervical, 1: Thoracic, 2: Lumbar
    c_start, c_end = get_region_bounds(mask_resized[0])
    t_start, t_end = get_region_bounds(mask_resized[1])
    l_start, l_end = get_region_bounds(mask_resized[2])
    
    # Combine masks for Shape Detection
    full_spine_mask = np.max(mask_resized, axis=0) # Merge all regions
    shape_label = determine_spine_shape(full_spine_mask)
    
    print("-" * 30)
    print("FINAL REPORT")
    print("-" * 30)
    print(f"View Type:      {view_label}")
    print(f"Spine Shape:    {shape_label}")
    print(f"Cervical Range: {c_start} to {c_end}")
    print(f"Thoracic Range: {t_start} to {t_end}")
    print(f"Lumbar Range:   {l_start} to {l_end}")
    
    # Visualize
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Original ({view_label})")
    
    plt.subplot(1,2,2)
    # Create RGB mask for display
    rgb_mask = np.zeros((original_h, original_w, 3))
    rgb_mask[..., 0] = mask_resized[0] # Red = Cervical
    rgb_mask[..., 1] = mask_resized[1] # Green = Thoracic
    rgb_mask[..., 2] = mask_resized[2] # Blue = Lumbar
    plt.imshow(rgb_mask)
    plt.title(f"AI Segmentation ({shape_label})")
    plt.show()

if __name__ == "__main__":
    # Test on a dummy image
    # analyze_patient("data/test_image.png")
    pass