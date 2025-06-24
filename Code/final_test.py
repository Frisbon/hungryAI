import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import InstrDataset, data_transforms
from model import OurCNN


BATCH_SIZE = 128
NUM_CLASSES = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_on_test_set():
    print(f"Using {device}")

    test_dataset = InstrDataset(
        csv_file='Dataset/test.csv',
        root_dir='Archive/instruments/test',
        transform=data_transforms['val']
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = OurCNN(num_classes=NUM_CLASSES).to(device)
    
    if not os.path.exists('Model/instrument_model.pth'):
        print("Error: Model file not found at Model/instrument_model.pth")
        return

    model.load_state_dict(torch.load('Model/instrument_model.pth'))
    loss_fn = nn.CrossEntropyLoss()
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(test_dataloader)
    correct /= len(test_dataloader.dataset)
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Accuracy on Test Set: {(100*correct):>0.1f}%")
    print(f"Average Loss on Test Set: {test_loss:>8f}")
    print("--------------------------")

if __name__ == '__main__':
    evaluate_on_test_set()