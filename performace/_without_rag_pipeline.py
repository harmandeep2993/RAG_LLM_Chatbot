import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as gr

# Load the FLAN-T5 large model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Check if a GPU is available and move the model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to generate chatbot responses
def chatbot_response(user_input):
    # Tokenize the user input
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # Generate a response using the model
    outputs = model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)

    # Decode the model output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio interface
def chat(user_input):
    return chatbot_response(user_input)

# Create a Gradio interface
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="FLAN-T5 Chatbot")

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
