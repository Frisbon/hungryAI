import os
import torch
from model import OurCNN # Or get_pretrained_model if you use that
from dataloader import data_transforms
import cv2

# --- START OF PATHING CHANGES ---
# This makes the script robust to being called from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

class_names_path = os.path.join(project_root, 'Dataset', 'class_names.txt')
model_path = os.path.join(project_root, 'Model', 'instrument_model.pth')
# --- END OF PATHING CHANGES ---


# --- 1. Load class names ---
try:
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f]
except FileNotFoundError:
    print(f"Error: class_names.txt not found at {class_names_path}")
    class_names = [f"Class {i}" for i in range(30)] # Fallback

NUM_CLASSES = len(class_names)


# --- 2. Initialize Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OurCNN(num_classes=NUM_CLASSES).to(device) # Or your best model architecture

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    # We can't proceed if the model doesn't exist, so we'll let the predict function handle this
    model = None

if model:
    model.eval()

# --- 3. Prediction Function ---
def predict_image(image_numpy):
    """
    Loads an image, preprocesses it, and returns the predicted class and confidence.
    """
    if model is None:
        return "Error: Model not loaded. Please train the model first.", 0.0

    try:
        # The input from Gradio is a NumPy array, so we use it directly
        image = data_transforms['val'](image_numpy).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, 1)
            
            predicted_class_name = class_names[predicted_class_idx.item()]
            confidence_score = confidence.item()

        return predicted_class_name, confidence_score

    except Exception as e:
        return f"An error occurred during prediction: {e}", 0.0