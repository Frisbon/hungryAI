# In Code/GUI.py

import gradio as gr
from predict import predict_image

def classify_food(image):
    """
    A wrapper function to connect the predictor to the Gradio interface.
    'image' here is a NumPy array from the Gradio component.
    """
    # Gradio provides a temp file path for the uploaded image
    predicted_class, confidence = predict_image(image)
    
    if "Error" in predicted_class:
        return predicted_class
    
    return {predicted_class: confidence}


# Create the Gradio interface
iface = gr.Interface(
    fn=classify_food,
    inputs=gr.Image(type="filepath", label="Upload a Food Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="GustoAI: Food Classifier",
    description="Upload an image of food, and this model will try to identify it. This is a project for the AI Lab course.",
    examples=[
        # Add paths to some example images from your test set if you like
    ]
)

# Launch the app
if __name__ == "__main__":
    iface.launch()