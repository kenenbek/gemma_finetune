"""
Model setup and management utilities for Gemma fine-tuning.
"""

import torch
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from config import ExperimentConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model and tokenizer setup with LoRA configuration."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

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


        # Load model with optimized settings for pipeline parallelism
        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
        }

        # Disable quantization for pipeline parallelism as it can cause issues
        if self.config.model.use_quantization and not self.config.model.use_pipeline_parallelism:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        elif self.config.model.use_quantization and self.config.model.use_pipeline_parallelism:
            logger.warning("Quantization disabled for pipeline parallelism to avoid compatibility issues")

        logger.info(f"Loading model with device_map: {device_map}")
        logger.info(f"Model kwargs: {list(model_kwargs.keys())}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name,
                **model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to load model with pipeline parallelism: {e}")
            logger.info("Falling back to single GPU loading...")
            # Fallback to single GPU
            fallback_kwargs = {
                "attn_implementation": self.config.model.attn_implementation,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.model_name,
                **fallback_kwargs
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
        elif self.config.model.use_minimal_training:
            # Minimal training mode: freeze almost all parameters for CPU debugging
            logger.info("Applying minimal training mode for CPU debugging")
            self._apply_minimal_training_freeze()
            logger.info("Model setup complete with minimal training configuration")
        else:
            # Full fine-tuning: make all parameters trainable
            for param in self.model.parameters():
                param.requires_grad = True

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Full fine-tuning mode: {trainable_params:,} trainable params out of {total_params:,} total params (100%)")
            logger.info("Model setup complete for full fine-tuning")

        return self.model
    
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

    def _apply_minimal_training_freeze(self):
        """
        Freeze almost all parameters, keeping only 0.001% trainable for CPU debugging.
        This method strategically selects a tiny subset of parameters to remain trainable.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before applying minimal training freeze")

        logger.info(f"Applying minimal training freeze (keeping {self.config.model.minimal_training_percent}% trainable)")

        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Count total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        target_trainable = max(1, int(total_params * (self.config.model.minimal_training_percent / 100)))

        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Target trainable parameters: {target_trainable:,}")

        # Strategy: Unfreeze only the final layer norm and a small portion of the last transformer layer
        # This gives us meaningful gradients while keeping the parameter count minimal
        trainable_count = 0

        # 1. Unfreeze final layer norm (small but important for output)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            for param in self.model.model.norm.parameters():
                if trainable_count + param.numel() <= target_trainable:
                    param.requires_grad = True
                    trainable_count += param.numel()
                    logger.info(f"Unfroze model.norm: {param.numel():,} parameters")

        # 2. Unfreeze bias terms from the last few transformer layers (if they exist and fit in budget)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Start from the last layer and work backwards
            for layer_idx in reversed(range(len(self.model.model.layers))):
                if trainable_count >= target_trainable:
                    break

                layer = self.model.model.layers[layer_idx]

                # Try to unfreeze bias parameters first (they're small but effective)
                for name, module in layer.named_modules():
                    if trainable_count >= target_trainable:
                        break
                    if hasattr(module, 'bias') and module.bias is not None:
                        if trainable_count + module.bias.numel() <= target_trainable:
                            module.bias.requires_grad = True
                            trainable_count += module.bias.numel()
                            logger.info(f"Unfroze layer {layer_idx}.{name}.bias: {module.bias.numel():,} parameters")

                # If we still have budget, unfreeze small weight matrices
                if trainable_count < target_trainable:
                    for name, param in layer.named_parameters():
                        if trainable_count >= target_trainable:
                            break
                        if param.requires_grad:  # Skip already unfrozen
                            continue
                        if trainable_count + param.numel() <= target_trainable:
                            # Prioritize smaller matrices like layer norms
                            if 'norm' in name.lower() or param.numel() < 1000:
                                param.requires_grad = True
                                trainable_count += param.numel()
                                logger.info(f"Unfroze layer {layer_idx}.{name}: {param.numel():,} parameters")

                # If we have enough trainable parameters, stop
                if trainable_count >= target_trainable * 0.8:  # Allow some flexibility
                    break

        # 3. If we still need more parameters and have budget, unfreeze lm_head bias
        if trainable_count < target_trainable and hasattr(self.model, 'lm_head'):
            if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None:
                if trainable_count + self.model.lm_head.bias.numel() <= target_trainable:
                    self.model.lm_head.bias.requires_grad = True
                    trainable_count += self.model.lm_head.bias.numel()
                    logger.info(f"Unfroze lm_head.bias: {self.model.lm_head.bias.numel():,} parameters")

        # Final count and validation
        actual_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        actual_percentage = (actual_trainable / total_params) * 100 if total_params > 0 else 0.0

        logger.info(f"✅ Minimal training setup complete:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {actual_trainable:,}")
        logger.info(f"  Percentage trainable: {actual_percentage:.4f}%")
        logger.info(f"  Frozen parameters: {total_params - actual_trainable:,}")

        # Verify we're within the target (with some tolerance)
        target_percent = self.config.model.minimal_training_percent
        if actual_percentage <= target_percent * 2:  # Allow 2x tolerance
            logger.info(f"✅ Successfully achieved minimal training target ({target_percent}%)")
        else:
            logger.warning(f"⚠️ Exceeded target by {actual_percentage - target_percent:.4f}%")

        # Ensure we have at least some trainable parameters
        if actual_trainable == 0:
            logger.warning("⚠️ No parameters were unfrozen! This might cause training issues.")
            # Force unfreeze at least one small parameter as fallback
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
                for param in self.model.model.norm.parameters():
                    param.requires_grad = True
                    logger.info(f"Fallback: Forced unfreezing of model.norm to ensure training can proceed")
                    break

        return actual_trainable, total_params
