"""
Model setup and management utilities for Gemma fine-tuning.
"""

import torch
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from config import ExperimentConfig
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model and tokenizer setup with LoRA configuration and Accelerate pipeline parallelism."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.accelerator = None
        self.pipeline_config = None

    def create_accelerate_pipeline_config(self) -> Dict[str, int]:
        """
        Create a balanced pipeline configuration for Gemma3-1B-IT using Accelerate.
        The model has 26 transformer layers (0-25) plus embeddings and head.
        IMPORTANT: Handles tied parameters (embed_tokens and lm_head) properly.
        """
        num_stages = self.config.model.num_pipeline_stages
        logger.info(f"Creating Accelerate pipeline config for {num_stages} stages")

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
        tied_param_stage = num_stages - 1

        # Strategy: Distribute layers evenly, then handle special components
        layers_per_stage = total_layers // num_stages
        remainder = total_layers % num_stages

        logger.info(f"Distributing {total_layers} layers across {num_stages} stages")
        logger.info(f"Base layers per stage: {layers_per_stage}, remainder: {remainder}")

        current_layer = 0

        # Distribute transformer layers first
        for stage in range(num_stages):
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
        logger.info("=== Final Accelerate Pipeline Configuration ===")
        for stage in range(num_stages):
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

        logger.info("✅ Accelerate pipeline configuration validation passed!")
        return pipeline_config

    def setup_accelerator(self) -> Accelerator:
        """Setup Accelerator with mixed precision for pipeline parallelism."""
        logger.info("Setting up Accelerator for pipeline parallelism")

        # Initialize accelerator with mixed precision
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.config.training.fp16 else "no",
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
        )

        logger.info(f"Accelerator initialized with {self.accelerator.num_processes} processes")
        logger.info(f"Using device: {self.accelerator.device}")

        return self.accelerator

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
        logger.info(f"Model: {self.config.model.model_name}")
        logger.info(f"Pipeline stages: {self.config.model.num_pipeline_stages}")

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

    def _create_balanced_device_map(self, num_gpus=4):
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
        logger.info("️IMPORTANT: Placing tied parameters (embed_tokens & lm_head) on same GPU to avoid conflicts")

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
                layers_on_this_gpu = max(1, layers_per_gpu - 2)  # At least 1 layer, but fewer than others
            else:
                # Other GPUs get their fair share plus some from the reserved space
                extra_layers = (layers_per_gpu + 1 - max(1, layers_per_gpu - 2)) // (num_gpus - 1)
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
            logger.error(
                f"❌ TIED PARAMETER VALIDATION FAILED: embed_tokens on GPU {embed_device}, lm_head on GPU {lm_head_device}")
            raise ValueError(
                f"Tied parameters must be on same device! embed_tokens: {embed_device}, lm_head: {lm_head_device}")
        else:
            logger.info(f"✅ TIED PARAMETER VALIDATION PASSED: Both on GPU {embed_device}")

        logger.info(f"Created balanced device map with tied parameter fix:")
        for gpu in range(num_gpus):
            layers_on_gpu = [key for key, device in device_map.items() if device == gpu and "layers" in key]
            other_components = [key for key, device in device_map.items() if device == gpu and "layers" not in key]
            logger.info(f"  GPU {gpu}: {len(layers_on_gpu)} layers + {other_components}")

        return device_map

    def setup_tokenizer(self):
        """Initialize the tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        return self.tokenizer

    def setup_model(self):
        """Initialize the model with optional LoRA configuration."""
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache")

        # Determine device mapping strategy
        device_map = None
        if self.config.model.use_pipeline_parallelism:
            logger.info(f"Pipeline parallelism enabled with {self.config.model.num_pipeline_stages} stages")
            logger.info(f"Device map strategy: {self.config.model.device_map_strategy}")

            device_map = self._create_balanced_device_map(num_gpus=self.config.model.num_pipeline_stages)

        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
        }

        logger.info(f"Loading model with device_map: {device_map}")
        logger.info(f"Model kwargs: {list(model_kwargs.keys())}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            **model_kwargs
        )

        if hasattr(self.model, 'hf_device_map'):
            logger.info(f"Final model device map: {self.model.hf_device_map}")

            # Log memory usage per GPU
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    if allocated > 0:
                        logger.info(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")

        # Apply LoRA only if use_peft is True
        if self.config.model.use_peft:
            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )

            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("Model setup complete with LoRA configuration")
        else:
            # Apply selective layer freezing if enabled
            if self.config.model.freeze_layers:
                logger.info("Applying selective layer freezing for Gemma model")
                self._apply_gemma_layer_freeze()

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            percentage_trainable = (trainable_params / total_params) * 100
            logger.info(f"Training mode: {trainable_params:,} trainable params out of {total_params:,} total params ({percentage_trainable:.2f}%)")
            logger.info("Model setup complete for fine-tuning with selective freezing" if self.config.model.freeze_layers else "Model setup complete for full fine-tuning")

        return self.model

    def setup_accelerate_pipeline_model(self):
        """Initialize the model using Accelerate pipeline parallelism."""
        logger.info("Setting up model with Accelerate pipeline parallelism")

        if not self.config.model.use_pipeline_parallelism:
            logger.warning("Pipeline parallelism not enabled in config, falling back to regular setup")
            return self.setup_model()

        # Setup accelerator
        self.accelerator = self.setup_accelerator()

        # Create pipeline configuration
        self.pipeline_config = self.create_accelerate_pipeline_config()

        # Initialize model and tokenizer with pipeline parallelism
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            device_map=self.pipeline_config,
            attn_implementation=self.config.model.attn_implementation,
            low_cpu_mem_usage=True,
        )

        logger.info("Model and tokenizer setup complete for Accelerate pipeline parallelism")

        # Log pipeline information
        self.log_pipeline_info()

        # Apply LoRA or layer freezing if needed
        if self.config.model.use_peft:
            logger.info("Applying LoRA to pipeline model")
            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                target_modules=self.config.lora.target_modules,
                lora_dropout=self.config.lora.lora_dropout,
                bias=self.config.lora.bias,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        elif self.config.model.freeze_layers:
            logger.info("Applying selective layer freezing to pipeline model")
            self._apply_gemma_layer_freeze()

        logger.info("Accelerate pipeline model setup complete")
        return self.model, self.accelerator

    def inference(self, text: str, max_new_tokens: int = 200) -> str:
        """Run inference on a single text."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        instruction = "Correct the spelling mistakes in the following Kyrgyz text:"
        prompt = f"<start_of_turn>user\n{instruction}\n{text}<end_of_turn>\n<start_of_turn>model\n"

        # Handle device placement for pipeline parallelism
        if self.config.model.use_pipeline_parallelism:
            # For pipeline parallelism, input should go to the first device
            first_device = 0 if torch.cuda.is_available() else "cpu"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(first_device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Use greedy decoding for deterministic results
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the corrected text
        try:
            corrected = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
            return corrected
        except:
            return response

    def _get_max_memory(self):
        """Get maximum memory allocation per GPU for pipeline parallelism."""
        if not torch.cuda.is_available():
            return None

        max_memory = {}
        num_gpus = min(torch.cuda.device_count(), self.config.model.num_pipeline_stages)

        for i in range(num_gpus):
            # Reserve some memory for training overhead and gradients
            # Use 85% of available memory per GPU
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_memory * 0.85)

        logger.info(f"Max memory allocation: {max_memory}")
        return max_memory

    def _get_conservative_max_memory(self):
        """Get conservative memory allocation per GPU for custom device mapping."""
        if not torch.cuda.is_available():
            return None

        conservative_max_memory = {}
        num_gpus = min(torch.cuda.device_count(), self.config.model.num_pipeline_stages)

        for i in range(num_gpus):
            # Reserve more memory for training overhead and gradients
            # Use 75% of available memory per GPU as a conservative estimate
            total_memory = torch.cuda.get_device_properties(i).total_memory
            conservative_max_memory[i] = int(total_memory * 0.75)

        logger.info(f"Conservative max memory allocation: {conservative_max_memory}")
        return conservative_max_memory

    def _apply_gemma_layer_freeze(self):
        """
        Apply selective layer freezing for Gemma3 model.
        Freezes everything except the last transformer layer (layer 25).
        """
        logger.info("🧊 Freezing everything except the last transformer layer of Gemma3 model...")

        # Freeze embeddings
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.model.model.rotary_emb.parameters():
            param.requires_grad = False
        for param in self.model.model.rotary_emb_local.parameters():
            param.requires_grad = False
        logger.info("❄️ Frozen embeddings (embed_tokens, rotary_emb, rotary_emb_local)")

        # Freeze LM head
        for param in self.model.lm_head.parameters():
            param.requires_grad = False
        logger.info("❄️ Frozen lm_head")

        # Freeze final layer norm
        for param in self.model.model.norm.parameters():
            param.requires_grad = False
        logger.info("❄️ Frozen final layer norm")

        # Freeze first 25 layers (0-24), keep layer 25 trainable
        for layer_idx in range(25):
            for param in self.model.model.layers[layer_idx].parameters():
                param.requires_grad = False
        logger.info("❄️ Frozen transformer layers 0-24")

        # Last layer (25) remains trainable by default
        trainable_params_last_layer = sum(p.numel() for p in self.model.model.layers[25].parameters())
        logger.info(f"🔥 Layer 25 remains trainable ({trainable_params_last_layer:,} parameters)")

        # Calculate statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        percentage_trainable = (trainable_params / total_params) * 100

        logger.info("🧊 ✅ Layer freezing complete!")
        logger.info(f"📊 Total: {total_params:,} | Trainable: {trainable_params:,} ({percentage_trainable:.2f}%)")

        return trainable_params, total_params
