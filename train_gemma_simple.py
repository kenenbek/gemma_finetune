"""
Simple training script for Gemma-3-1B-IT for Kyrgyz spell checking.
Uses LoRA without quantization to avoid bitsandbytes issues.
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os
import json
import logging
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from data import KyrgyzDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleTrainingConfig:
    """Configuration for training parameters."""
    model_name: str = "google/gemma-3-1b-it"
    dataset_path: str = "../misspelled_kg_dataset/"
    output_dir: str = "./kyrgyz_gemma_spellcheck_simple"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_length: int = 256
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    num_samples: int = 64  # Use small subset for testing

    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


class SimpleKyrgyzDataset(Dataset):
    """Simple dataset class for Kyrgyz spell checking."""

    def __init__(self, misspelled_texts: List[str], correct_texts: List[str], tokenizer, max_length: int = 256):
        self.misspelled = misspelled_texts
        self.correct = correct_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.misspelled)

    def __getitem__(self, idx):
        misspelled = self.misspelled[idx]
        correct = self.correct[idx]

        # Simple format for instruction tuning
        text = f"Fix: {misspelled} -> {correct}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "labels": encoded["input_ids"].flatten()
        }


def main():
    """Main training function."""
    config = SimpleTrainingConfig()

    try:
        logger.info("="*50)
        logger.info("STARTING KYRGYZ SPELL CHECK TRAINING")
        logger.info("="*50)
        logger.info(f"Configuration: {config}")

        logger.info("Step 1/7: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✓ Tokenizer loaded successfully. Vocab size: {len(tokenizer)}")

        logger.info("Step 2/7: Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info(f"✓ Model loaded successfully. Model size: {model.num_parameters():,} parameters")

        logger.info("Step 3/7: Setting up LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("✓ LoRA setup complete")

        logger.info("Step 4/7: Loading data...")
        loader = KyrgyzDataLoader()
        splits = loader.load_from_hf_dataset(
            config.dataset_path,
            num_samples=config.num_samples
        )

        # Create datasets
        train_dataset = SimpleKyrgyzDataset(
            splits['train'][0], splits['train'][1],
            tokenizer, config.max_length
        )

        val_dataset = SimpleKyrgyzDataset(
            splits['val'][0], splits['val'][1],
            tokenizer, config.max_length
        )

        logger.info(f"✓ Data loaded successfully:")
        logger.info(f"  Train dataset size: {len(train_dataset)}")
        logger.info(f"  Val dataset size: {len(val_dataset)}")

        logger.info("Step 5/7: Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=True,
            report_to=None,
        )
        logger.info("✓ Training arguments configured")

        logger.info("Step 6/7: Initializing trainer...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        logger.info("✓ Trainer initialized")

        logger.info("Step 7/7: Starting training...")
        logger.info("="*50)
        trainer.train()

        logger.info("Training completed! Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)

        logger.info("✓ Model saved successfully!")
        logger.info("="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()
