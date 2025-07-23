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
    
    def _create_pipeline_device_map(self):
        """Create custom device map for pipeline parallelism."""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"

        num_gpus = min(torch.cuda.device_count(), self.config.model.num_pipeline_stages)
        logger.info(f"Setting up pipeline parallelism across {num_gpus} GPUs")

        # Check GPU memory
        for i in range(num_gpus):
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {memory_gb:.1f}GB total memory")

        if num_gpus == 1:
            return {"": 0}

        # For Gemma-3-1B, which has 26 layers (0-25)
        total_layers = 26
        layers_per_gpu = total_layers // num_gpus
        remainder = total_layers % num_gpus

        device_map = {}

        # Embedding layers on first GPU
        device_map["model.embed_tokens"] = 0
        device_map["model.rotary_emb"] = 0
        device_map["model.rotary_emb_local"] = 0

        # Distribute transformer layers more evenly
        current_layer = 0
        for gpu_id in range(num_gpus):
            # Add one extra layer to first 'remainder' GPUs
            layers_on_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)

            for _ in range(layers_on_this_gpu):
                if current_layer < total_layers:
                    device_map[f"model.layers.{current_layer}"] = gpu_id
                    current_layer += 1

        # Final layers on last GPU
        device_map["model.norm"] = num_gpus - 1
        device_map["lm_head"] = num_gpus - 1

        logger.info(f"Created custom device map for {num_gpus} GPUs:")
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
        if self.config.model.use_pipeline_parallelism:
            logger.info(f"Pipeline parallelism enabled with {self.config.model.num_pipeline_stages} stages")
            logger.info(f"Device map strategy: {self.config.model.device_map_strategy}")

            # Force custom mapping for better control
            if self.config.model.device_map_strategy in ["balanced", "auto"]:
                logger.info("Forcing custom device mapping due to HuggingFace allocation issues")
                device_map = self._create_pipeline_device_map()
            elif self.config.model.device_map_strategy == "custom":
                device_map = self._create_pipeline_device_map()
            else:
                device_map = self._create_pipeline_device_map()
        else:
            device_map = "auto"

        # Load model with optimized settings for pipeline parallelism
        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation,
            "torch_dtype": torch.float16,  # Use fp16 for better memory efficiency
            "low_cpu_mem_usage": True,  # Important for multi-GPU loading
        }

        # Add device_map only if it's not a string (i.e., custom mapping)
        if isinstance(device_map, dict):
            model_kwargs["device_map"] = device_map
            # Set a more conservative max_memory for custom mapping
            model_kwargs["max_memory"] = self._get_conservative_max_memory()
        else:
            model_kwargs["device_map"] = device_map

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

