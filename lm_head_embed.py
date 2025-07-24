import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use a smaller model for a quick download
model_id = "google/gemma-3-1b-it"
print(f"Loading model: {model_id}")

# Load the model onto the CPU to inspect it
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

print(tokenizer.pad_token)
print(tokenizer.pad_token)