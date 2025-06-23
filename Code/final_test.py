import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader import FoodDataset, data_transforms
from model import OurCNN # Or get_pretrained_model if you use that

# --- PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# --- NOTE: These paths point to the TEST data ---
csv_test_path = os.path.join(project_root, 'Dataset', 'test.csv')
test_image_root_path = os.path.join(project_root, 'Archive', 'instruments', 'test')
model_path = os.path.join(project_root, 'Model', 'instrument_model.pth')

# --- CONFIGURATION ---
BATCH_SIZE = 128
NUM_CLASSES = 30 # Make sure this matches your dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_on_test_set():
    """Loads the best model and evaluates it on the final test set."""
    print(f"Using {device} device for final evaluation.")

    # Load the test dataset
    test_dataset = FoodDataset(
        csv_file=csv_test_path,
        root_dir=test_image_root_path,
        transform=data_transforms['val'] # Use validation transforms (no augmentation)
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Load the trained model
    model = OurCNN(num_classes=NUM_CLASSES).to(device) # Initialize your final model architecture
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train your model using train_test.py first.")
        return

    print(f"Loading saved model state from {model_path}")
    model.load_state_dict(torch.load(model_path))

    # Evaluate the model
    loss_fn = nn.CrossEntropyLoss()
    size = len(test_dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    print("\n--- FINAL TEST RESULTS ---")
    print(f"Accuracy on Test Set: {(100*correct):>0.1f}%")
    print(f"Average Loss on Test Set: {test_loss:>8f}")
    print("--------------------------")

if __name__ == '__main__':
    evaluate_on_test_set()