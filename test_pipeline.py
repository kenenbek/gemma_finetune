#!/usr/bin/env python3
"""
Test script to verify pipeline parallelism setup before running full training.
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_custom_device_map(num_gpus=4):
    """Create custom device map for Gemma-3-1B."""
    # For Gemma-3-1B, which has 26 layers (0-25)
    total_layers = 26
    layers_per_gpu = total_layers // num_gpus
    remainder = total_layers % num_gpus

    device_map = {}

    # Place embedding and lm_head on the same GPU to avoid tied parameter issues
    # We'll put them on the last GPU along with the final layers
    last_gpu = num_gpus - 1

    # Rotary embeddings on first GPU
    device_map["model.rotary_emb"] = 0
    device_map["model.rotary_emb_local"] = 0

    # Distribute transformer layers, leaving more room on last GPU for embeddings and lm_head
    current_layer = 0
    for gpu_id in range(num_gpus):
        if gpu_id == last_gpu:
            # Last GPU gets fewer transformer layers to make room for embeddings and lm_head
            layers_on_this_gpu = total_layers - current_layer
        else:
            layers_on_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)

        for _ in range(layers_on_this_gpu):
            if current_layer < total_layers:
                device_map[f"model.layers.{current_layer}"] = gpu_id
                current_layer += 1

    # Place embeddings and language modeling head on the same GPU (last GPU)
    device_map["model.embed_tokens"] = last_gpu
    device_map["model.norm"] = last_gpu
    device_map["lm_head"] = last_gpu

    logger.info(f"Created custom device map for {num_gpus} GPUs (embeddings and lm_head on same GPU):")
    for gpu in range(num_gpus):
        layers_on_gpu = [key for key, device in device_map.items() if device == gpu and "layers" in key]
        other_components = [key for key, device in device_map.items() if device == gpu and "layers" not in key]
        logger.info(f"  GPU {gpu}: {len(layers_on_gpu)} layers + {other_components}")

    return device_map

def test_pipeline_parallelism():
    """Test pipeline parallelism setup."""
    logger.info("Testing pipeline parallelism setup...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")
    
    # Log GPU info
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f}GB")
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Test custom device mapping strategy
    logger.info(f"\n=== Testing custom device mapping ===")

    try:
        # Create custom device map
        custom_device_map = create_custom_device_map(min(4, num_gpus))

        # Set conservative max memory per GPU (75% of total)
        max_memory = {}
        for i in range(min(4, num_gpus)):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_memory * 0.75)

        logger.info(f"Conservative max memory per GPU: {max_memory}")

        # Load model with custom device mapping
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            device_map=custom_device_map,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            attn_implementation="eager"
        )

        logger.info(f"Model loaded successfully with custom device mapping!")
        logger.info(f"Device map: {model.hf_device_map}")

        # Check if model is actually distributed
        devices_used = set(model.hf_device_map.values())
        logger.info(f"Model distributed across {len(devices_used)} devices: {devices_used}")

        # Check memory usage
        for i in range(num_gpus):
            if torch.cuda.memory_allocated(i) > 0:
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB used ({allocated/total*100:.1f}%)")

        # Test a simple forward pass
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_input = "Hello, this is a test."
        inputs = tokenizer(test_input, return_tensors="pt")

        # Move inputs to first device (GPU 0)
        inputs = {k: v.to(0) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logger.info(f"Forward pass successful! Output shape: {outputs.logits.shape}")

        # Test generation
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            logger.info(f"Generation test successful! Generated: '{generated_text[:100]}...'")

        # Clean up
        del model
        torch.cuda.empty_cache()
        logger.info(f"Custom device mapping test completed successfully!")

        return True

    except Exception as e:
        logger.error(f"Failed with custom device mapping: {e}")
        torch.cuda.empty_cache()
        return False

if __name__ == "__main__":
    success = test_pipeline_parallelism()
    if success:
        logger.info("✅ Pipeline parallelism test PASSED!")
    else:
        logger.error("❌ Pipeline parallelism test FAILED!")
