import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import SpineDataset
from src.models import UNetSegmentor
from src.loss import DiceBCELoss
from src.config import *

def train():
    # 1. Setup Data
    train_ds = SpineDataset(
        images_dir=f"{DATA_DIR}/raw_images",
        masks_dir=f"{DATA_DIR}/processed_masks"
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Setup Model, Loss, Optimizer
    model = UNetSegmentor(n_classes=NUM_SEG_CLASSES).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler() # Mixed precision for speed

    print("--- Starting Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        epoch_loss = 0
        
        for idx, (data, targets) in enumerate(loop):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Step Scheduler
        scheduler.step()

        # Save Checkpoint
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/region_segmentor.pth")
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader)}")

if __name__ == "__main__":
    train()