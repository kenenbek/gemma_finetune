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
            attn_implementation="eager"
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
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-7,
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




def main():
    """Main function to run the training pipeline."""
    # Set GPU visibility to use only GPU 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Create configuration
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
