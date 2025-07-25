"""
Main script for training Gemma-3-1B-IT for Kyrgyz spell checking.
Uses modular components for better code organization.
"""

import os
import logging
from config import ExperimentConfig, ModelConfig, LoRAConfig, DataConfig, TrainingConfig
from trainer import KyrgyzSpellCheckTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_config():
    """Create the default experiment configuration."""
    return ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=256,
            use_quantization=False,
            attn_implementation="eager",
            use_peft=True  # Set to False for full fine-tuning
        ),
        lora=LoRAConfig(
            r=32,  # Increased rank for better capacity
            lora_alpha=64,  # Increased alpha (typically 2x rank)
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Reduced dropout for better learning
            bias="none"
        ),
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=4096,  # Small number for testing
            max_val_samples=512,  # Limit validation dataset size
            max_length=256
        ),
        training=TrainingConfig(
            output_dir="./kyrgyz_spellcheck_model",
            num_train_epochs=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-7,  # Lower LR for full fine-tuning, higher for LoRA
            weight_decay=0.01,
            warmup_steps=50,
            logging_steps=25,
            save_steps=200,
            eval_steps=20,
            save_total_limit=3,
            fp16=True,
            eval_accumulation_steps=32,
            use_wandb=True,
            run_name="kyrgyz-spellcheck-gemma"
        )
    )


def create_full_finetuning_config():
    """Create configuration for full fine-tuning (without PEFT)."""
    return ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=256,
            use_quantization=False,
            attn_implementation="eager",
            use_peft=False,  # Disable PEFT for full fine-tuning
            # Enable pipeline parallelism for 4-GPU cluster
            use_pipeline_parallelism=False,
            num_pipeline_stages=4,
            device_map_strategy="custom"
        ),
        lora=LoRAConfig(),  # Still needed for config structure but won't be used
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=4,  # Use full dataset for better training
            max_val_samples=2,
            max_length=256
        ),
        training=TrainingConfig(
            output_dir="./spellcheck_model_full_pipeline",
            num_train_epochs=1,
            per_device_train_batch_size=1,  # Slightly larger batch size with pipeline parallelism
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,  # Reduced due to larger batch size
            learning_rate=5e-7,  # Much lower learning rate for full fine-tuning
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=-1,
            save_steps=2,
            eval_steps=2,
            save_total_limit=4,
            fp16=False,
            eval_accumulation_steps=32,
            use_wandb=True,
            run_name="spellcheck-gemma-full-pipeline"
        )
    )


def create_cpu_debug_config():
    """Create configuration for CPU debugging with minimal training."""
    return ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=128,  # Shorter sequences for faster CPU processing
            use_quantization=False,  # Disable quantization for CPU
            attn_implementation="eager",
            use_peft=False,  # Disable PEFT
            use_pipeline_parallelism=False,  # Disable pipeline parallelism for CPU
            use_minimal_training=True,  # Enable minimal training
            minimal_training_percent=0.00001  # Only 0.001% of parameters trainable
        ),
        lora=LoRAConfig(),  # Still needed for config structure but won't be used
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=1,  # Very small dataset for quick debugging
            max_val_samples=1,  # Small validation set
            max_length=128
        ),
        training=TrainingConfig(
            output_dir="./debug_model_cpu",
            num_train_epochs=2,  # Just a few epochs for testing
            per_device_train_batch_size=1,  # Small batch size for CPU
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,  # Higher learning rate since we're training very few parameters
            weight_decay=0.01,
            warmup_steps=5,  # Few warmup steps
            logging_steps=1,  # Log every step for debugging
            save_steps=5,
            eval_steps=5,
            save_total_limit=2,
            fp16=False,  # Disable fp16 for CPU
            eval_accumulation_steps=1,
            use_wandb=False,  # Disable wandb for debugging
            run_name="cpu-debug-minimal-training"
        )
    )


def main():
    """Main function to run the training pipeline."""
    # Choose configuration type
    # Set use_cpu_debug = True for CPU debugging with minimal training
    # Set use_full_finetuning = True to train without PEFT
    # Set use_cpu_debug = False and use_full_finetuning = False for default LoRA training

    use_cpu_debug = False  # Set to True for CPU debugging
    use_full_finetuning = True

    if use_cpu_debug:
        logger.info("Using CPU debugging configuration with minimal training (0.001% parameters)")
        config = create_cpu_debug_config()
    elif use_full_finetuning:
        logger.info("Using full fine-tuning configuration (no PEFT)")
        config = create_full_finetuning_config()
    else:
        logger.info("Using PEFT (LoRA) configuration")
        config = create_default_config()

    # Initialize trainer
    trainer = KyrgyzSpellCheckTrainer(config)

    try:
        # Run training/evaluation
        trainer.train()

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
