import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify, render_template

# Set up logging to print in real-time without duplication
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create cache directory if it doesn't exist
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)
logging.info(f"Cache directory set to: {cache_dir}")

# Hugging Face token (ensure you are authorized to use the model)
hf_token = "auth"

# Specify the model name (ensure you have access to this model)
model_name = "meta-llama/Llama-3.2-3B-Instruct"
logging.info(f"Using model: {model_name}")

# Check for available GPU and set precision accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    logging.warning("CUDA is not available. Using CPU.")

# Load tokenizer
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)

# Load the model with CPU offloading in case of GPU out of memory
try:
    if torch.cuda.is_available():
        logging.info("Loading model with FP16 precision for GPU with CPU offloading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            token=hf_token, 
            cache_dir=cache_dir, 
            torch_dtype=torch.float16, 
            device_map="auto",
            max_memory={0: "2.5GiB", "cpu": "10GiB"}  # Changed 'cuda:0' to 0 for proper memory management
        )
        logging.info("Model loaded with FP16 precision and CPU offloading.")
    else:
        logging.info("Loading model with standard precision for CPU...")
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir).to(device)
        logging.info("Model loaded with standard precision on CPU.")
except torch.cuda.OutOfMemoryError:
    logging.error("CUDA out of memory! Offloading model to CPU.")
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir).to("cpu")
    logging.info("Model loaded entirely on CPU due to insufficient GPU memory.")

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Pad token set to EOS token: {tokenizer.pad_token}")

app = Flask(__name__)

# Home route to display the chat interface
@app.route("/", methods=["GET"])
def home():
    logging.info("Rendering home page (chat interface)")
    return render_template("index.html")

# API route for generating text
@app.route("/generate-text", methods=["POST"])
def generate_text():
    logging.info("Received text generation request")
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        logging.warning("No prompt provided in the request")
        return jsonify({'error': 'Please provide a prompt.'}), 400

    logging.info(f"Prompt received: {prompt}")

    # Tokenize the prompt and move input tensors to the appropriate device (GPU/CPU)
    logging.info("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    logging.info(f"Tokenized input ids: {inputs.input_ids}")

    # Increase max_length for generation; adjust as needed
    max_length = min(32768, len(inputs.input_ids[0]) + 32768)  # Adjusted max length to 512 tokens
    logging.info(f"Max length for generation set to: {max_length}")

    # Generate text using faster generation settings (e.g., FP16 on CUDA if available)
    logging.info("Generating text...")
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            attention_mask=inputs.attention_mask,
            do_sample=True,  # Sampling for more creative responses
            temperature=0.7,  # Adjust for more random/creative generation
            top_p=0.9,  # Use nucleus sampling for diversity
            pad_token_id=tokenizer.eos_token_id,
        )

    logging.info(f"Generated output ids: {outputs}")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Generated text: {generated_text}")

    # Format the generated text for better readability (e.g., break into paragraphs)
    formatted_text = format_text(generated_text)
    logging.info(f"Formatted text: {formatted_text}")

    return jsonify({'generated_text': formatted_text})

# Helper function to format text (e.g., splitting into paragraphs)
def format_text(text):
    # Split the text into sentences for better display in the chat
    sentences = text.split(". ")
    formatted_text = "<br>".join(sentences)  # Add line breaks between sentences
    logging.info("Text formatted with line breaks for display")
    return formatted_text

if __name__ == "_main_":
    logging.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)