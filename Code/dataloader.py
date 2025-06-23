import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class FoodDataset(Dataset): # You can rename this to InstrumentDataset if you like
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# --- START OF CHANGES ---
# Define separate transforms for training (with augmentation) and validation (without)
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images
        transforms.RandomRotation(15),         # Randomly rotate images
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Adjust color
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
}