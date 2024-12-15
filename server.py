from flask import Flask, request, jsonify, render_template
from q_gpt import GPT
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging  # Add logging import

 

app = Flask(__name__)


print(torch.__version__)
print(torch.cuda.is_available())

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

# Load the model
model = None
enc = None

def initialize_model():
    global model, enc
    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to('cuda')
    enc = tiktoken.get_encoding('gpt2')

def generate_text(model, enc, input_text, num_return_sequences=5, max_length=30):
    # Tokenize input text
    tokens = enc.encode(input_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to('cuda')

    # Generate text
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    while tokens.size(1) < max_length:
        with torch.no_grad():
            logits = model(tokens)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            tokens = torch.cat((tokens, xcol), dim=1)

    # Decode and return the generated text
    generated_texts = []
    for i in range(num_return_sequences):
        decoded = enc.decode(tokens[i, :max_length].tolist())
        generated_texts.append(decoded)
    
    return generated_texts

@app.route("/") 
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_text_api():
    data = request.json
    input_text = data.get('input_text', '')  # Get the chat message from the UI
    
    logging.debug(f"Received input_text: {input_text}")  # Log the received input text

    # Initialize the model if not already initialized
    if model is None:
        logging.info("Initializing the model...")  # Log model initialization
        initialize_model()
    
    # Call the generation function
    generated_texts = generate_text(model, enc, input_text)  # Generate text based on the chat message
    
    logging.debug(f"Generated texts: {generated_texts}")  # Log the generated texts
    
    return jsonify({'generated_texts': generated_texts})

if __name__ == "__main__":
    app.run(debug=True)