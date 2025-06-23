# In Code/GUI.py

import gradio as gr
from predict import predict_image

def classify_food(image_numpy):
    """
    A wrapper function to connect the predictor to the Gradio interface.
    'image' here is a NumPy array from the Gradio component.
    """
    # The predict_image function expects a file path, so we save the numpy array
    # from Gradio as a temporary file and pass the path.
    # Note: Gradio handles the temporary file cleanup.
    predicted_class, confidence = predict_image(image_numpy)
    
    if "Error" in predicted_class:
        return predicted_class
    
    # Format the output for Gradio's Label component
    return {predicted_class: confidence}


# Create the Gradio interface
iface = gr.Interface(
    fn=classify_food,
    # Input is now a NumPy array, which is more direct for the predict function
    inputs=gr.Image(type="numpy", label="Upload a Food Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="GustoAI: Food Classifier",
    description="Upload an image of food, and this model will try to identify it. This is a project for the AI Lab course.",
    allow_flagging="never" # Disables the flagging button
)

# Launch the app
if __name__ == "__main__":
    iface.launch()