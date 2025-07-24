#!/usr/bin/env python3
"""
Test script to verify balanced pipeline parallelism setup for Gemma-3 models.
This version uses a balanced device mapping strategy that distributes components
more evenly across GPUs for better memory utilization.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable TensorFloat32 for better performance on A10G and other modern GPUs
# This fixes the TF32 warning and provides better performance
torch.set_float32_matmul_precision('high')
logger.info("✅ TensorFloat32 enabled for better GPU performance")


def create_balanced_device_map(num_gpus=4):
    """
    Create balanced device map for Gemma-3-1B.
    This strategy distributes components more evenly across GPUs for better balance.
    IMPORTANT: Places tied parameters (embed_tokens and lm_head) on the same GPU.
    """
    logger.info(f"Creating balanced device map for {num_gpus} GPUs...")

    # For Gemma-3-1B, which has 26 layers (0-25)
    total_layers = 26
    layers_per_gpu = total_layers // num_gpus
    remainder = total_layers % num_gpus

    device_map = {}

    # CRITICAL: In Gemma models, embed_tokens and lm_head share weights (tied parameters)
    # They MUST be on the same device to avoid the tied parameter conflict
    logger.info(f"Total layers: {total_layers}, layers per GPU: {layers_per_gpu}, remainder: {remainder}")
    logger.info("⚠️  IMPORTANT: Placing tied parameters (embed_tokens & lm_head) on same GPU to avoid conflicts")

    # Strategy: Place both embeddings and lm_head on the last GPU
    # This leaves more room on other GPUs for transformer layers
    tied_param_gpu = num_gpus - 1  # Use last GPU for tied parameters

    # Place embeddings and related components on the tied parameter GPU
    device_map["model.embed_tokens"] = tied_param_gpu
    device_map["model.rotary_emb"] = tied_param_gpu
    device_map["model.norm"] = tied_param_gpu
    device_map["lm_head"] = tied_param_gpu

    # Also place rotary_emb_local if it exists (this fixes the CUDA graph warning)
    device_map["model.rotary_emb_local"] = tied_param_gpu

    logger.info(f"Placing tied parameters on GPU {tied_param_gpu}:")
    logger.info(f"  - model.embed_tokens -> GPU {tied_param_gpu}")
    logger.info(f"  - lm_head -> GPU {tied_param_gpu}")
    logger.info(f"  - model.norm -> GPU {tied_param_gpu}")
    logger.info(f"  - model.rotary_emb -> GPU {tied_param_gpu}")
    logger.info(f"  - model.rotary_emb_local -> GPU {tied_param_gpu}")

    # Distribute transformer layers across all GPUs, but reserve space on last GPU
    # Since the last GPU has the heavy embedding/lm_head components, give it fewer layers
    current_layer = 0
    assigned_layers = []  # Track which layers we've assigned

    for gpu_id in range(num_gpus):
        if gpu_id == tied_param_gpu:
            # Last GPU gets fewer layers since it has embeddings + lm_head
            layers_on_this_gpu = max(1, layers_per_gpu - 1)  # At least 1 layer, but fewer than others
        else:
            # Other GPUs get their fair share plus some from the reserved space
            extra_layers = (layers_per_gpu + 1 - max(1, layers_per_gpu - 1)) // (num_gpus - 1)
            layers_on_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0) + extra_layers

        layer_start = current_layer
        gpu_layers = []  # Track layers assigned to this GPU

        for _ in range(layers_on_this_gpu):
            if current_layer < total_layers:
                device_map[f"model.layers.{current_layer}"] = gpu_id
                assigned_layers.append(current_layer)
                gpu_layers.append(current_layer)
                current_layer += 1

        logger.info(f"GPU {gpu_id}: layers {gpu_layers} ({len(gpu_layers)} layers)")

    # Assign any remaining layers to the first few GPUs
    remaining_gpu = 0
    while current_layer < total_layers:
        if remaining_gpu != tied_param_gpu:  # Don't overload the tied parameter GPU
            device_map[f"model.layers.{current_layer}"] = remaining_gpu
            assigned_layers.append(current_layer)
            logger.info(f"GPU {remaining_gpu}: added remaining layer {current_layer}")
            current_layer += 1
        remaining_gpu = (remaining_gpu + 1) % num_gpus

    # VALIDATION: Ensure all layers are assigned
    expected_layers = list(range(total_layers))
    assigned_layers_sorted = sorted(assigned_layers)

    logger.info(f"Validation: Expected layers: {expected_layers}")
    logger.info(f"Validation: Assigned layers: {assigned_layers_sorted}")

    if assigned_layers_sorted == expected_layers:
        logger.info("✅ VALIDATION PASSED: All layers properly assigned!")
    else:
        missing_layers = set(expected_layers) - set(assigned_layers)
        extra_layers = set(assigned_layers) - set(expected_layers)

        if missing_layers:
            logger.error(f"❌ VALIDATION FAILED: Missing layers: {sorted(missing_layers)}")
        if extra_layers:
            logger.error(f"❌ VALIDATION FAILED: Extra layers: {sorted(extra_layers)}")

        raise ValueError(f"Layer assignment validation failed! Missing: {missing_layers}, Extra: {extra_layers}")

    # Additional validation: Check for duplicates
    if len(assigned_layers) != len(set(assigned_layers)):
        duplicates = [layer for layer in assigned_layers if assigned_layers.count(layer) > 1]
        logger.error(f"❌ VALIDATION FAILED: Duplicate layer assignments: {duplicates}")
        raise ValueError(f"Duplicate layer assignments found: {duplicates}")

    # Validate tied parameters are on same device
    embed_device = device_map["model.embed_tokens"]
    lm_head_device = device_map["lm_head"]
    if embed_device != lm_head_device:
        logger.error(f"❌ TIED PARAMETER VALIDATION FAILED: embed_tokens on GPU {embed_device}, lm_head on GPU {lm_head_device}")
        raise ValueError(f"Tied parameters must be on same device! embed_tokens: {embed_device}, lm_head: {lm_head_device}")
    else:
        logger.info(f"✅ TIED PARAMETER VALIDATION PASSED: Both on GPU {embed_device}")

    logger.info(f"Created balanced device map with tied parameter fix:")
    for gpu in range(num_gpus):
        layers_on_gpu = [key for key, device in device_map.items() if device == gpu and "layers" in key]
        other_components = [key for key, device in device_map.items() if device == gpu and "layers" not in key]
        logger.info(f"  GPU {gpu}: {len(layers_on_gpu)} layers + {other_components}")

    return device_map

def analyze_memory_requirements():
    """Analyze memory requirements before loading the model."""
    logger.info("Analyzing GPU memory state...")

    # Clear cache first
    torch.cuda.empty_cache()

    # Log initial memory state
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        cached = torch.cuda.memory_reserved(i) / (1024**3)
        free = total_memory - allocated - cached
        logger.info(f"GPU {i}: {free:.1f}GB free, {allocated:.1f}GB allocated, {cached:.1f}GB cached")

def test_pipeline_parallelism():
    """Test pipeline parallelism setup."""
    logger.info("Testing balanced pipeline parallelism setup...")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")

    # Log GPU info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024 ** 3)
        logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f}GB")

    # Analyze memory before starting
    analyze_memory_requirements()

    # Test balanced device mapping strategy
    logger.info(f"\n=== Testing balanced device mapping ===")

    try:
        # Create ACTUAL balanced device map (not just a string)
        balanced_device_map = create_balanced_device_map(min(4, num_gpus))
        logger.info(f"Balanced device map created: {balanced_device_map}")

        # Set memory limits - be more conservative to avoid allocation issues
        max_memory = {}
        for i in range(min(4, num_gpus)):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            # Use 70% instead of 75% to be more conservative
            max_memory[i] = int(total_memory * 0.70)

        logger.info(f"Max memory per GPU (70% of total): {max_memory}")

        # Log memory in GB for readability
        for i, mem in max_memory.items():
            mem_gb = mem / (1024**3)
            logger.info(f"  GPU {i}: {mem_gb:.1f}GB max allowed")

        logger.info("Loading model with balanced device mapping...")

        # Load model with ACTUAL balanced device mapping (not string "balanced")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            device_map=balanced_device_map,  # Use actual device map, not string
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            attn_implementation="eager"
        )

        logger.info(f"Model loaded successfully with balanced device mapping!")
        logger.info(f"Actual device map used by model: {model.hf_device_map}")

        # Check if model is actually distributed
        devices_used = set(model.hf_device_map.values())
        logger.info(f"Model distributed across {len(devices_used)} devices: {devices_used}")

        # Verify the mapping matches our expectation
        if len(devices_used) > 1:
            logger.info("✅ Model successfully distributed across multiple GPUs!")
        else:
            logger.warning("⚠️ Model ended up on single GPU despite balanced mapping attempt")

        # Check memory usage after model loading
        logger.info("Memory usage after model loading:")
        gpu_usage = {}
        total_memory_used = 0
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
            if allocated > 0 or cached > 0:
                usage_pct = allocated/total*100
                gpu_usage[i] = usage_pct
                total_memory_used += allocated
                logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached / {total:.1f}GB total ({usage_pct:.1f}%)")

        logger.info(f"Total memory used across all GPUs: {total_memory_used:.1f}GB")

        # Calculate balance metrics if multiple GPUs are used
        if len(gpu_usage) > 1:
            avg_usage = sum(gpu_usage.values()) / len(gpu_usage)
            max_usage = max(gpu_usage.values())
            min_usage = min(gpu_usage.values())
            balance_score = 100 - ((max_usage - min_usage) / avg_usage * 100) if avg_usage > 0 else 0

            logger.info(f"Memory Balance Analysis:")
            logger.info(f"  Average usage: {avg_usage:.1f}%")
            logger.info(f"  Max usage: {max_usage:.1f}%")
            logger.info(f"  Min usage: {min_usage:.1f}%")
            logger.info(f"  Balance score: {balance_score:.1f}% (higher is better)")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare test input
        test_input = "Hello, this is a test of balanced pipeline parallelism."
        inputs = tokenizer(test_input, return_tensors="pt")
        logger.info(f"Test input: '{test_input}'")
        logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")

        # Determine where to send inputs based on where embeddings are located
        embed_device = model.hf_device_map.get("model.embed_tokens", 0)
        logger.info(f"Embeddings are located on device: {embed_device}")
        logger.info(f"Moving inputs to device {embed_device}...")

        # Move inputs to the device where embeddings are located
        inputs = {k: v.to(embed_device) for k, v in inputs.items()}
        logger.info(f"Inputs successfully moved to device {embed_device}")

        # Test forward pass
        logger.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(f"Forward pass successful! Output shape: {outputs.logits.shape}")
            logger.info(f"Output tensor is on device: {outputs.logits.device}")

        # Test generation
        logger.info("Testing text generation...")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"Generation successful!")
            logger.info(f"Generated text: '{generated_text}'")

        # Final memory check
        logger.info("Final memory usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            if allocated > 0:
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")

        # Clean up
        del model
        torch.cuda.empty_cache()
        logger.info(f"Balanced device mapping test completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Failed with balanced device mapping: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())

        # Log which device each component ended up on for debugging
        try:
            if 'model' in locals():
                logger.error(f"Model device map at time of error: {model.hf_device_map}")
        except:
            pass

        torch.cuda.empty_cache()
        return False

if __name__ == "__main__":
    success = test_pipeline_parallelism()
    if success:
        logger.info("✅ Balanced pipeline parallelism test PASSED!")
    else:
        logger.error("❌ Balanced pipeline parallelism test FAILED!")
