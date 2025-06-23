# In Code/train_test.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import FoodDataset, data_transforms
from model import OurCNN

# --- START OF PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_train_path = os.path.join(project_root, 'Dataset', 'train.csv')
csv_val_path = os.path.join(project_root, 'Dataset', 'validation.csv')
image_root_path = os.path.join(project_root, 'Archive', 'all_images')
model_save_path = os.path.join(project_root, 'Model', 'food_cnn_model.pth')
# --- END OF PATHING SETUP ---

# -- 1. Hyperparameters and Setup --
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 34

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(dataloader, model, loss_fn, optimizer, scaler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            pred = model(X)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # --- START OF THE FIX ---
    # This line tells PyTorch to use a more compatible algorithm for matrix multiplication
    torch.backends.cuda.matmul.allow_tf32 = False
    # --- END OF THE FIX ---

    print(f"Using {device} device")

    train_dataset = FoodDataset(
        csv_file=csv_train_path,
        root_dir=image_root_path,
        transform=data_transforms
    )
    validation_dataset = FoodDataset(
        csv_file=csv_val_path,
        root_dir=image_root_path,
        transform=data_transforms
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4
    )

    model = OurCNN(num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists(model_save_path):
        print(f"Loading saved model state from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        print("Resuming training...")
    else:
        print("No saved model found, starting training from scratch.")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, scaler)
        test_loop(validation_dataloader, model, loss_fn)
        # In Code/train_test.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import FoodDataset, data_transforms
from model import OurCNN

# --- START OF PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_train_path = os.path.join(project_root, 'Dataset', 'train.csv')
csv_val_path = os.path.join(project_root, 'Dataset', 'validation.csv')
image_root_path = os.path.join(project_root, 'Archive', 'all_images')
model_save_path = os.path.join(project_root, 'Model', 'food_cnn_model.pth')
# --- END OF PATHING SETUP ---

# -- 1. Hyperparameters and Setup --
LEARNING_RATE = 1e-1
BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 34

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(dataloader, model, loss_fn, optimizer, scaler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            pred = model(X)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # ==================================================================
    # THE FIX: This line avoids the CUBLAS error on some systems
    torch.backends.cuda.matmul.allow_tf32 = False
    # ==================================================================

    print(f"Using {device} device")

    train_dataset = FoodDataset(
        csv_file=csv_train_path,
        root_dir=image_root_path,
        transform=data_transforms
    )
    validation_dataset = FoodDataset(
        csv_file=csv_val_path,
        root_dir=image_root_path,
        transform=data_transforms
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = OurCNN(num_classes=NUM_CLASSES).to(device)

    if os.path.exists(model_save_path):
        print(f"Loading saved model state from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        print("Resuming training...")
    else:
        print("No saved model found, starting training from scratch.")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler
    print("Done!")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved PyTorch Model State to {model_save_path}")