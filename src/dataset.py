import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.config import IMG_SIZE

class SpineDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)
        
        # Define default augmentation if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE),
                A.CLAHE(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])
        
        # Assumes mask has same filename as image
        mask_path = os.path.join(self.masks_dir, self.images[index]) 

        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Load Mask (Expects 3 Channels: R=Cervical, G=Thoracic, B=Lumbar)
        # If mask doesn't exist (e.g. for classification only), return zeros
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path) 
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB) # Ensure RGB
        else:
            mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # 3. Apply Transformations
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # Normalize mask to 0-1 float
        mask = mask.float() / 255.0

        return image, mask