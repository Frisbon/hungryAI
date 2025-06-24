import torch
from model import OurCNN 
from dataloader import data_transforms

# loading class names from a text file
try:
    with open('Dataset/class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f]
except FileNotFoundError:
    print("Error: txt not found")
    class_names = [f"Class {i}" for i in range(30)] 

NUM_CLASSES = len(class_names)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = OurCNN(num_classes=NUM_CLASSES).to(device) 

try:
    model.load_state_dict(torch.load('Model/instrument_model.pth', map_location=device))
except FileNotFoundError:
    print("Error: Model file not found")
    model = None

if model:
    model.eval() 

# loads an image, preprocesses it, and returns the predicted class (str) and confidence % (float)
def predict_image(image_numpy):
    if model is None:
        return "Error: Ao train the model first.", 0.0

    try:
        # input from Gradio is a NumPy array
        image = data_transforms['val'](image_numpy).unsqueeze(0).to(device) 

        with torch.no_grad(): 
            outputs = model(image)
            probTensor = torch.nn.functional.softmax(outputs, dim=1)
            confScore, predicted_class_idx = torch.max(probTensor, 1) 
            
            predicted_class_name = class_names[predicted_class_idx.item()]
            confidence_score = confScore.item()

        return predicted_class_name, confidence_score

    except Exception as _:
        return "Error during prediction", 0.0