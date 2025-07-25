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


def create_lora_config():
    """Create the default experiment configuration."""
    return ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=256,
            attn_implementation="eager",
            use_peft=True
        ),
        lora=LoRAConfig(
            r=8,  # Increased rank for better capacity
            lora_alpha=16,  # Increased alpha (typically 2x rank)
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,  # Reduced dropout for better learning
            bias="none"
        ),
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=4,  # Small number for testing
            max_val_samples=1,  # Limit validation dataset size
            max_length=256
        ),
        training=TrainingConfig(
            output_dir="./kyrgyz_spellcheck_model",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            learning_rate=2e-7,  # Lower LR for full fine-tuning, higher for LoRA
            weight_decay=0.01,
            warmup_steps=1,
            logging_steps=2,
            save_steps=2,
            eval_steps=2,
            save_total_limit=1,
            fp16=True,
            eval_accumulation_steps=1,
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
            attn_implementation="eager",
            use_peft=False,  # Disable PEFT for full fine-tuning
            # Enable pipeline parallelism for 4-GPU cluster
            use_pipeline_parallelism=False,
            num_pipeline_stages=0,
            device_map_strategy="custom",
            freeze_layers=True
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


def create_accelerate_pipeline_config():
    """Create configuration for Accelerate pipeline parallelism."""
    return ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=256,
            attn_implementation="eager",
            use_peft=False,  # Can be True for LoRA with pipeline
            # Enable Accelerate pipeline parallelism
            use_pipeline_parallelism=True,
            use_accelerate_pipeline=True,
            num_pipeline_stages=4,
            device_map_strategy="accelerate",
            freeze_layers=True
        ),
        lora=LoRAConfig(),  # Still needed for config structure
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=4,
            max_val_samples=2,
            max_length=256
        ),
        training=TrainingConfig(
            output_dir="./spellcheck_model_accelerate_pipeline",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-7,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=-1,  # Disable console logging
            save_steps=2,
            eval_steps=2,
            save_total_limit=4,
            fp16=True,  # Use fp16 for better memory efficiency
            eval_accumulation_steps=32,
            use_wandb=True,
            run_name="spellcheck-gemma-accelerate-pipeline"
        )
    )


def main():
    """Main function to run the training pipeline."""
    # Choose configuration type
    # Set use_accelerate_pipeline = True to use Accelerate pipeline parallelism
    # Set use_full_finetuning = True to train without PEFT
    # Set use_cpu_debug = False and use_full_finetuning = False for default LoRA training

    use_accelerate_pipeline = True  # NEW: Use Accelerate pipeline parallelism
    use_full_finetuning = False

    if use_accelerate_pipeline:
        logger.info("Using Accelerate pipeline parallelism configuration")
        config = create_accelerate_pipeline_config()
    elif use_full_finetuning:
        logger.info("Using full fine-tuning configuration (no PEFT)")
        config = create_full_finetuning_config()
    else:
        logger.info("Using PEFT (LoRA) configuration")
        config = create_lora_config()

    # Initialize trainer
    trainer = KyrgyzSpellCheckTrainer(config)

    try:
        trainer.train()

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
