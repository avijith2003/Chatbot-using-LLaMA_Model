import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model and authentication details
model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = "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"  

# Define the cache directory for storing the model and tokenizer
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# Download and cache the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, cache_dir=cache_dir)

print("Model and tokenizer downloaded successfully.")

