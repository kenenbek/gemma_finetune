"""
Accelerate-based pipeline parallelism for Gemma3-1B-IT model.
Provides balanced layer distribution across multiple GPUs.
"""

import logging
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AcceleratePipelineManager:
    """Manages pipeline parallelism using Accelerate for Gemma3-1B-IT."""
    
    def __init__(self, model_name: str = "google/gemma-3-1b-it", num_stages: int = 4):
        self.model_name = model_name
        self.num_stages = num_stages
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        
    def create_balanced_pipeline_config(self) -> Dict[str, int]:
        """
        Create a balanced pipeline configuration for Gemma3-1B-IT.
        The model has 26 transformer layers (0-25) plus embeddings and head.
        IMPORTANT: Handles tied parameters (embed_tokens and lm_head) properly.
        """
        logger.info(f"Creating balanced pipeline config for {self.num_stages} stages")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU-only device mapping")
            # For CPU, put everything on device 0
            pipeline_config = {}
            pipeline_config["model.embed_tokens"] = 0
            pipeline_config["model.rotary_emb"] = 0
            pipeline_config["lm_head"] = 0  # Keep tied parameters on same device
            pipeline_config["model.norm"] = 0

            # All layers on device 0 for CPU
            for layer_idx in range(26):
                pipeline_config[f"model.layers.{layer_idx}"] = 0

            return pipeline_config

        # GPU pipeline configuration
        total_layers = 26  # layers 0-25
        pipeline_config = {}

        # CRITICAL: Place tied parameters (embed_tokens and lm_head) on the SAME device
        # We'll use the last stage for both to avoid conflicts
        tied_param_stage = self.num_stages - 1

        # Strategy: Distribute layers evenly, then handle special components
        layers_per_stage = total_layers // self.num_stages
        remainder = total_layers % self.num_stages

        logger.info(f"Distributing {total_layers} layers across {self.num_stages} stages")
        logger.info(f"Base layers per stage: {layers_per_stage}, remainder: {remainder}")

        current_layer = 0

        # Distribute transformer layers first
        for stage in range(self.num_stages):
            layers_in_this_stage = layers_per_stage
            if stage < remainder:
                layers_in_this_stage += 1

            stage_layers = []
            for _ in range(layers_in_this_stage):
                if current_layer < total_layers:
                    pipeline_config[f"model.layers.{current_layer}"] = stage
                    stage_layers.append(current_layer)
                    current_layer += 1
                    
            logger.info(f"Stage {stage}: layers {stage_layers} ({len(stage_layers)} layers)")

        # Verify all layers are assigned
        if current_layer != total_layers:
            logger.error(f"Layer assignment error: assigned {current_layer} layers, expected {total_layers}")
            raise ValueError(f"Not all layers assigned! Assigned: {current_layer}, Expected: {total_layers}")

        # Now assign special components
        # Place embeddings on first stage
        pipeline_config["model.embed_tokens"] = 0
        pipeline_config["model.rotary_emb"] = 0

        # Place norm and lm_head on last stage (tied with embed_tokens)
        pipeline_config["model.norm"] = tied_param_stage
        pipeline_config["lm_head"] = tied_param_stage

        # CRITICAL FIX: Move embed_tokens to the same device as lm_head for tied parameters
        pipeline_config["model.embed_tokens"] = tied_param_stage
        logger.info(f"🔧 TIED PARAMETER FIX: Moving embed_tokens to stage {tied_param_stage} (same as lm_head)")

        # Log final configuration
        logger.info("=== Final Pipeline Configuration ===")
        for stage in range(self.num_stages):
            stage_components = [comp for comp, s in pipeline_config.items() if s == stage]
            layer_components = [comp for comp in stage_components if "model.layers." in comp]
            other_components = [comp for comp in stage_components if "model.layers." not in comp]

            logger.info(f"Stage {stage}:")
            logger.info(f"  - Layers: {len(layer_components)} ({[int(comp.split('.')[2]) for comp in layer_components if comp.startswith('model.layers.')]})")
            logger.info(f"  - Other: {other_components}")

        # Validation: Check all required components are present
        required_components = ["model.embed_tokens", "model.rotary_emb", "model.norm", "lm_head"]
        required_components.extend([f"model.layers.{i}" for i in range(26)])

        missing_components = [comp for comp in required_components if comp not in pipeline_config]
        if missing_components:
            logger.error(f"Missing components in device map: {missing_components}")
            raise ValueError(f"Missing components: {missing_components}")

        logger.info("✅ Pipeline configuration validation passed!")
        return pipeline_config
        
    def setup_accelerator(self) -> Accelerator:
        """Setup Accelerator with mixed precision."""
        logger.info("Setting up Accelerator for pipeline parallelism")

        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision="fp16",  # Use fp16 for better memory efficiency
            gradient_accumulation_steps=1,
        )
        
        logger.info(f"Accelerator initialized with {self.accelerator.num_processes} processes")
        logger.info(f"Using device: {self.accelerator.device}")

        return self.accelerator
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with pipeline parallelism."""
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with device_map for pipeline parallelism
        pipeline_config = self.create_balanced_pipeline_config()
        
        # Convert to device map format expected by transformers
        device_map = {}
        for component, stage in pipeline_config.items():
            device_map[component] = stage
            
        # Adjust model loading parameters based on device availability
        model_kwargs = {
            "device_map": device_map,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }

        # Only use torch_dtype=float16 if CUDA is available
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("Using fp16 precision for GPU")
        else:
            logger.info("Using default precision for CPU")

        logger.info("Loading model with pipeline device map")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare model with accelerator (this handles the distributed setup)
        if self.accelerator:
            # Note: For pipeline parallelism, we don't need to call prepare_model
            # as the device_map already handles the distribution
            logger.info("Model prepared for pipeline parallelism")

        logger.info("Model and tokenizer setup complete")
        return self.model, self.tokenizer
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics for each GPU."""
        if not torch.cuda.is_available():
            return {}
            
        stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            
            stats[f"GPU_{i}"] = {
                "allocated_GB": allocated,
                "reserved_GB": reserved,
                "total_GB": total,
                "utilization_%": (allocated / total) * 100 if total > 0 else 0
            }
            
        return stats
        
    def log_pipeline_info(self):
        """Log pipeline configuration and memory usage."""
        logger.info("=== Pipeline Parallelism Configuration ===")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Pipeline stages: {self.num_stages}")
        
        if hasattr(self.model, 'hf_device_map'):
            logger.info("Device map:")
            for component, device in self.model.hf_device_map.items():
                logger.info(f"  {component} -> GPU {device}")
                
        # Memory statistics
        memory_stats = self.get_memory_stats()
        if memory_stats:
            logger.info("\nMemory usage:")
            for gpu, stats in memory_stats.items():
                logger.info(f"  {gpu}: {stats['allocated_GB']:.1f}GB allocated "
                           f"({stats['utilization_%']:.1f}% of {stats['total_GB']:.1f}GB)")


class AccelerateTrainerWrapper:
    """Wrapper for SFTTrainer to work with Accelerate pipeline parallelism."""
    
    def __init__(self, pipeline_manager: AcceleratePipelineManager):
        self.pipeline_manager = pipeline_manager
        self.accelerator = pipeline_manager.accelerator
        
    def prepare_training_components(self, trainer, train_dataloader, eval_dataloader, optimizer, scheduler):
        """Prepare training components with Accelerate."""
        logger.info("Preparing training components with Accelerate")
        
        # Prepare components
        model, optimizer, train_dataloader, eval_dataloader, scheduler = self.accelerator.prepare(
            trainer.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler
        )
        
        # Update trainer with prepared components
        trainer.model = model
        trainer.optimizer = optimizer
        trainer.lr_scheduler = scheduler
        
        return trainer
        
    def backward(self, loss):
        """Backward pass with Accelerate."""
        self.accelerator.backward(loss)
        
    def clip_grad_norm(self, max_norm: float):
        """Clip gradients with Accelerate."""
        self.accelerator.clip_grad_norm_(self.pipeline_manager.model.parameters(), max_norm)
        
    def wait_for_everyone(self):
        """Synchronize all processes."""
        self.accelerator.wait_for_everyone()
        
    def save_state(self, output_dir: str):
        """Save model state using Accelerate."""
        self.accelerator.save_state(output_dir)
        
    def load_state(self, input_dir: str):
        """Load model state using Accelerate."""
        self.accelerator.load_state(input_dir)


def create_accelerate_config_file(num_gpus: int = 4, output_path: str = "accelerate_config.yaml"):
    """
    Create an Accelerate configuration file for pipeline parallelism.
    
    Args:
        num_gpus: Number of GPUs to use
        output_path: Path to save the config file
    """
    config_content = f"""compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: {num_gpus}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    
    with open(output_path, 'w') as f:
        f.write(config_content)
        
    logger.info(f"Accelerate config saved to {output_path}")
    logger.info("To use this config, run: accelerate launch --config_file accelerate_config.yaml your_script.py")
    
    return output_path


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create pipeline manager
    pipeline_manager = AcceleratePipelineManager(
        model_name="google/gemma-3-1b-it",
        num_stages=4
    )
    
    # Setup accelerator
    accelerator = pipeline_manager.setup_accelerator()
    
    # Setup model and tokenizer
    model, tokenizer = pipeline_manager.setup_model_and_tokenizer()
    
    # Log pipeline information
    pipeline_manager.log_pipeline_info()
    
    # Create accelerate config file
    create_accelerate_config_file(num_gpus=4)
    
    logger.info("Pipeline parallelism setup complete!")
