import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template, jsonify

torch.cuda.empty_cache()

# Set up environment variables to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  
# Initialize Flask app
app = Flask(__name__)

# Model details
model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = "hugging_face_auth_token"

# Define the cache directory for storing the model and tokenizer
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Load tokenizer and model with float16
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)

# Attempt mixed precision (half precision)
torch.cuda.empty_cache()  # Clear cache
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir).to("cuda")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question", "")

    # Tokenize input question
    inputs = tokenizer(question, return_tensors="pt").to("cuda")

    with torch.cuda.amp.autocast():  # Use mixed precision
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
