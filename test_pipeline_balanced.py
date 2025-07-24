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
        memory_gb = props.total_memory / (1024 ** 3)
        logger.info(f"GPU {i}: {props.name}, {memory_gb:.1f}GB")

    # Clear cache
    torch.cuda.empty_cache()

    # Test custom device mapping strategy
    logger.info(f"\n=== Testing balanced device mapping ===")

    try:
        # Set conservative max memory per GPU (75% of total)
        max_memory = {}
        for i in range(min(4, num_gpus)):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_memory * 0.75)

        logger.info(f"Conservative max memory per GPU: {max_memory}")

        # Load model with custom device mapping
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            device_map="balanced",
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
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB used ({allocated / total * 100:.1f}%)")

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


