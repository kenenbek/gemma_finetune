import torch
from transformers import AutoModelForCausalLM

# Use a smaller model for a quick download
model_id = "google/gemma-3-1b-it"
print(f"Loading model: {model_id}")

# Load the model onto the CPU to inspect it
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model loaded.")

# 1. Get the input embedding layer and its weight tensor
input_embeddings_layer = model.get_input_embeddings()
input_weights = input_embeddings_layer.weight
print(f"\nInput Embeddings Layer: {input_embeddings_layer}")
print(f"Input Weights Shape: {input_weights.shape}")

# 2. Get the output projection layer (the LM head) and its weight tensor
output_embeddings_layer = model.get_output_embeddings()
output_weights = output_embeddings_layer.weight
print(f"\nOutput Embeddings (LM Head) Layer: {output_embeddings_layer}")
print(f"Output Weights Shape: {output_weights.shape}")

# 3. The Ultimate Proof: Check their memory addresses
input_id = id(input_weights)
output_id = id(output_weights)

print(f"\nMemory ID of input_weights:  {input_id}")
print(f"Memory ID of output_weights: {output_id}")

if input_id == output_id:
    print("\n✅ SUCCESS: The memory IDs are identical. They are the SAME tensor object.")
else:
    print("\n❌ FAILURE: The memory IDs are different.")

# 4. Another Proof: The 'is' operator checks for object identity
if input_weights is output_weights:
    print("✅ SUCCESS: Python's 'is' operator confirms they are the same object.")
else:
    print("❌ FAILURE: The 'is' operator shows they are different objects.")

# 5. The "Destructive" Proof: Change one and see if the other changes
# Let's change a value in the input weights and check the output weights
print("\n--- Performing destructive test ---")
original_value = output_weights[10, 10].item()
print(f"Original value at output_weights[10, 10]: {original_value}")

# Change the value in the INPUT weights
with torch.no_grad():
    input_weights[10, 10] = 999.99

new_value = output_weights[10, 10].item()
print(f"New value at output_weights[10, 10] after modifying input_weights: {new_value}")

if new_value != original_value:
    print("✅ SUCCESS: Modifying the input weights directly changed the output weights!")
else:
    print("❌ FAILURE: The weights are independent.")