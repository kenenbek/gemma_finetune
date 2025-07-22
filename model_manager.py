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

        # Load model with quantization if specified
        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation,
            "device_map": "auto"
        }

        if self.config.model.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            **model_kwargs
        )
        print(self.model.hf_device_map)

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
