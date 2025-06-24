import gradio as gr
from predict import predict_image

def classify_instrument(image_numpy):
    predicted_class, confidence = predict_image(image_numpy)
    if "Error" in predicted_class:
        return predicted_class
    return {predicted_class: confidence}

custom_css = """
body {
    font-family: "Segoe UI", "Helvetica", sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
    background-color: #E7C7A3 !important;
}

.gradio-container {
    background-color: #E7C7A3 !important;
}

h1 {
    font-weight: 800 !important;
    font-size: 2.5rem !important;
    text-align: center !important;
    color: #6B3406 !important;
    margin-top: 1em !important;
    margin-bottom: 0.2em !important;
}

h2 {
    text-align: center !important;
    color: #99775b !important;
    font-size: 1.1rem !important;
}

/* Blocchi input/output */
.custom-box {
    background-color: #f6e9d7 !important;
    border: #6b3406 10px !important;
    border-radius: 12px !important;
    padding: 20px !important;
    height: 100% !important; /* Riempie tutta la colonna */
    box-sizing: border-box !important;
}

/* Bottone centrato */
.centered-button {
    display: flex !important;
    justify-content: center !important;
    justify-items: center !important;
    justify-self: center !important;
    align-items: center !important;
    align-content: center !important;
    align-self: center !important;
    margin-top: 30px !important;
    width: 60% !important;
}

.gr-button {
    background: #6b3406 !important;
    background-color: #6b3406 !important;
    color: #fffcf8 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    border: none !important;
}

.gr-button:hover {
    background-color: #99775b !important;
    background: #99775b !important;
}
"""


with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as GUI:
    gr.HTML("""
        <h1>🎵 InstrumAI: Instrument Classifier</h1>
        <h2>Upload an image of a musical instrument and let the AI identify it!</h2>
    """)

    with gr.Row(equal_height=True):  
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="🎼 Upload Instrument Image", elem_classes="custom-box")
        with gr.Column(scale=1):
            result = gr.Label(num_top_classes=1, label="🎧 Predicted Instrument", elem_classes="custom-box")

    with gr.Row(elem_classes="centered-button"):
        submit_btn = gr.Button("Classify Instrument 🎶")

    submit_btn.click(fn=classify_instrument, inputs=image_input, outputs=result)

if __name__ == "__main__":
    GUI.launch()