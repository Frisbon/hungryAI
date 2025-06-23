# In Code/predict.py

import torch
from model import OurCNN
from dataloader import data_transforms
import cv2

# --- 1. Load class names ---
class_names = []
with open('../Dataset/class_names.txt', 'r') as f:
    for line in f:
        class_names.append(line.strip())

# --- 2. Initialize Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OurCNN(num_classes=len(class_names))
model.load_state_dict(torch.load('../Model/food_cnn_model.pth', map_location=device))
model.to(device)
model.eval()

# --- 3. Prediction Function ---
def predict_image(image_path):
    """
    Loads an image, preprocesses it, and returns the predicted class and confidence.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Image not found or could not be opened.", 0.0
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply the same transformations as during training
        transformed_image = data_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(transformed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            
            predicted_class_name = class_names[predicted_class_idx.item()]
            confidence_score = confidence.item()

        return predicted_class_name, confidence_score

    except Exception as e:
        return f"An error occurred: {e}", 0.0