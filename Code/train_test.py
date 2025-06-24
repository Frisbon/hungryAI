import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import InstrDataset, data_transforms
from model import OurCNN

# HYPERPARAMETERS :)
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_loop(dataloader, model, loss_fn, optimizer, scaler):
    model.train()
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            pred = model(X)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"Using {device}")

    train_dataset = InstrDataset(
        csv_file='Dataset/train.csv',
        root_dir='Archive/instruments/train',
        transform=data_transforms['train']
    )
    validation_dataset = InstrDataset(
        csv_file='Dataset/validation.csv',
        root_dir='Archive/instruments/valid',
        transform=data_transforms['val']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = OurCNN(num_classes=NUM_CLASSES).to(device)
    
    if os.path.exists('Model/instrument_model.pth'):
        print("Loading saved model...")
        model.load_state_dict(torch.load('Model/instrument_model.pth'))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, scaler)
        test_loop(validation_dataloader, model, loss_fn)
        
    print("the end!")
    torch.save(model.state_dict(), 'Model/instrument_model.pth')
    print("Saved PyTorch Model State to Model/instrument_model.pth")