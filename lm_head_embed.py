import torch
from transformers import AutoModelForCausalLM

# Use a smaller model for a quick download
model_id = "google/gemma-3-1b-it"
print(f"Loading model: {model_id}")

# Load the model onto the CPU to inspect it
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model loaded.")

print(model)