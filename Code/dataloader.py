# In Code/dataloader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 # OpenCV for image loading

class FoodDataset(Dataset):
    """Custom Dataset for loading food images."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and label from the dataframe
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformations for the images
# We resize them and convert them to tensors
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])