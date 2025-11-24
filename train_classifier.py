import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import sys

# Import project config and model
from src.config import *
from src.models import SpineViewClassifier

def get_transforms(mode='train'):
    """
    Standard ImageNet normalization.
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(), # Augmentation
            transforms.RandomRotation(10),     # Augmentation
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Handle different X-ray exposures
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train():
    # --- 1. SETUP DATA ---
    data_path = os.path.join(DATA_DIR, "train_classifier")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"ERROR: Data directory not found at {data_path}")
        print("Please create folders: data/train_classifier/AP and data/train_classifier/Lateral")
        return

    # Use ImageFolder: Automatically assigns label 0 to first folder, 1 to second
    full_dataset = datasets.ImageFolder(root=data_path, transform=get_transforms('train'))
    
    # Simple Split: 80% Train, 20% Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Overwrite validation transform (remove augmentation)
    # Note: A cleaner way is separate folders, but this is quick for mixed data
    val_dataset.dataset.transform = get_transforms('val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Classes Detected: {full_dataset.class_to_idx}")
    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images")

    # --- 2. SETUP MODEL ---
    model = SpineViewClassifier().to(DEVICE)
    
    # CrossEntropyLoss automatically handles the softmax calculation
    criterion = nn.CrossEntropyLoss()
    
    # Adam Optimizer with weight decay (L2 regularization) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning Rate Scheduler: Reduces LR if accuracy stops improving
    # Learning Rate Scheduler: Cosine Annealing with Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # --- 3. TRAINING LOOP ---
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
            loop.set_postfix(loss=loss.item())

        epoch_loss = train_loss / len(train_dataset)
        epoch_acc = train_correct.double() / len(train_dataset)

        # --- 4. VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)

        print(f"Epoch {epoch+1} Results: Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
        # Step Scheduler
        # Step Scheduler
        scheduler.step()

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(CHECKPOINT_DIR, "view_classifier.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Best Accuracy! Model Saved to {save_path}")

    print("Training Complete.")

if __name__ == "__main__":
    # Create checkpoint dir if not exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train()