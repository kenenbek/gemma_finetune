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
    
    # Test different device mapping strategies
    strategies = ["balanced", "auto"]
    
    for strategy in strategies:
        logger.info(f"\n=== Testing {strategy} strategy ===")
        
        try:
            # Set max memory per GPU (85% of total)
            max_memory = {}
            for i in range(min(4, num_gpus)):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                max_memory[i] = int(total_memory * 0.85)
            
            logger.info(f"Max memory per GPU: {max_memory}")
            
            # Load model with pipeline parallelism
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-1b-it",
                device_map=strategy,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                attn_implementation="eager"
            )
            
            logger.info(f"Model loaded successfully with {strategy} strategy!")
            logger.info(f"Device map: {model.hf_device_map}")
            
            # Check memory usage
            for i in range(num_gpus):
                if torch.cuda.memory_allocated(i) > 0:
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB used")
            
            # Test a simple forward pass
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            test_input = "Hello, this is a test."
            inputs = tokenizer(test_input, return_tensors="pt")
            
            # Move inputs to first device
            first_device = next(iter(model.hf_device_map.values()))
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logger.info(f"Forward pass successful! Output shape: {outputs.logits.shape}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            logger.info(f"{strategy} strategy test completed successfully!\n")
            
        except Exception as e:
            logger.error(f"Failed with {strategy} strategy: {e}")
            torch.cuda.empty_cache()
            continue
    
    return True

if __name__ == "__main__":
    test_pipeline_parallelism()
