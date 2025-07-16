"""
Finetuning procedure for Gemma-3-1B-IT for Kyrgyz spell checking.
Uses LoRA (Low-Rank Adaptation) for efficient finetuning.
"""

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import os
import json
import logging
from typing import List, Optional, Literal
from dataclasses import dataclass
import wandb
import jiwer
import numpy as np
from data import KyrgyzDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "google/gemma-3-1b-it"
    max_length: int = 512
    use_quantization: bool = False
    attn_implementation: str = "eager"


@dataclass
class LoRAConfig:
    """Configuration for LoRA settings."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class DataConfig:
    """Configuration for dataset settings."""
    dataset_path: str = "../misspelled_kg_dataset/"
    num_samples: Optional[int] = 512
    max_length: int = 512


@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    output_dir: str = "./kyrgyz_spellcheck_model"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 2
    save_total_limit: int = 3
    fp16: bool = True
    eval_accumulation_steps: int = 1024
    use_wandb: bool = False
    run_name: str = "kyrgyz-spellcheck-gemma"


@dataclass
class ExperimentConfig:
    """Main configuration class that combines all configs."""
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {}))
        )

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "lora": self.lora.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__
        }


class KyrgyzSpellCheckDataset(Dataset):
    """Dataset class for Kyrgyz spell checking formatted for instruction tuning."""

    def __init__(self, misspelled_texts: List[str], correct_texts: List[str], tokenizer, max_length: int = 512):
        self.misspelled = misspelled_texts
        self.correct = correct_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.misspelled)

    def __getitem__(self, idx):
        misspelled = self.misspelled[idx]
        correct = self.correct[idx]

        # Format as instruction-following task
        instruction = "Correct the spelling mistakes in the following Kyrgyz text:"
        prompt = f"<start_of_turn>user\n{instruction}\n{misspelled}<end_of_turn>\n<start_of_turn>model\n{correct}<end_of_turn>"

        # Tokenize
        encoded = self.tokenizer(
            prompt,
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


class KyrgyzSpellCheckTrainer:
    """Trainer class for Kyrgyz spell checking model."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup_tokenizer(self):
        """Initialize the tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def setup_model(self):
        """Initialize the model with LoRA configuration."""
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Load model with quantization if specified
        model_kwargs = {
            "attn_implementation": self.config.model.attn_implementation
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

    def load_data(self):
        """Load and prepare datasets."""
        logger.info("Loading Kyrgyz spell checking dataset...")

        # Load data using the existing data loader
        loader = KyrgyzDataLoader()
        splits = loader.load_from_hf_dataset(
            self.config.data.dataset_path,
            num_samples=self.config.data.num_samples
        )

        # Create datasets
        self.train_dataset = KyrgyzSpellCheckDataset(
            splits['train'][0], splits['train'][1],
            self.tokenizer, self.config.data.max_length
        )

        self.val_dataset = KyrgyzSpellCheckDataset(
            splits['val'][0], splits['val'][1],
            self.tokenizer, self.config.data.max_length
        )

        self.test_dataset = KyrgyzSpellCheckDataset(
            splits['test'][0], splits['test'][1],
            self.tokenizer, self.config.data.max_length
        )

        logger.info(f"Datasets loaded:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Validation: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")

    def setup_training_arguments(self):
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=self.config.training.save_total_limit,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=self.config.training.fp16,
            eval_accumulation_steps=self.config.training.eval_accumulation_steps,
            prediction_loss_only=True,
            report_to="wandb" if self.config.training.use_wandb else None,
            run_name=self.config.training.run_name
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Extract corrected text from model outputs
        corrected_texts = []
        reference_texts = []

        for pred, label in zip(decoded_preds, decoded_labels):
            # Extract the corrected text after the model response
            try:
                pred_text = pred.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
                label_text = label.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
                corrected_texts.append(pred_text)
                reference_texts.append(label_text)
            except:
                corrected_texts.append(pred)
                reference_texts.append(label)

        # Calculate metrics
        exact_match = sum(1 for p, r in zip(corrected_texts, reference_texts) if p == r) / len(corrected_texts)

        # Calculate character-level accuracy
        char_accuracy = []
        for pred, ref in zip(corrected_texts, reference_texts):
            if len(ref) > 0:
                char_acc = 1 - jiwer.cer(ref, pred)
                char_accuracy.append(max(0.0, char_acc))
            else:
                char_accuracy.append(1.0 if len(pred) == 0 else 0.0)

        # Calculate word error rate (WER)
        wer_scores = []
        for pred, ref in zip(corrected_texts, reference_texts):
            if len(ref.strip()) > 0:
                wer = jiwer.wer(ref, pred)
                wer_scores.append(wer)
            else:
                wer_scores.append(0.0 if len(pred.strip()) == 0 else 1.0)

        return {
            "exact_match": exact_match,
            "character_accuracy": np.mean(char_accuracy),
            "word_error_rate": np.mean(wer_scores)
        }

    def train(self):
        """Main training function."""
        logger.info("Starting training setup...")

        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        self.load_data()

        # Initialize wandb if enabled
        if self.config.training.use_wandb:
            wandb.init(
                project="kyrgyz-spellcheck",
                config=self.config.to_dict(),
                name=f"gemma-kyrgyz-spellcheck-{self.config.training.num_train_epochs}epochs"
            )

        # Setup training arguments
        training_args = self.setup_training_arguments()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        # Test evaluation
        test_results = trainer.evaluate(eval_dataset=self.test_dataset)
        logger.info(f"Test results: {test_results}")

        # Save training config
        with open(os.path.join(self.config.training.output_dir, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        if self.config.training.use_wandb:
            wandb.log({"final_eval": eval_results, "test_results": test_results})
            wandb.finish()

        logger.info("Training completed successfully!")

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
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the corrected text
        try:
            corrected = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
            return corrected
        except:
            return response


def main():
    config = ExperimentConfig(
        model=ModelConfig(
            model_name="google/gemma-3-1b-it",
            max_length=512,
            use_quantization=False,
            attn_implementation="eager"
        ),
        lora=LoRAConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none"
        ),
        data=DataConfig(
            dataset_path="../misspelled_kg_dataset/",
            num_samples=512,
            max_length=512
        ),
        training=TrainingConfig(
            output_dir="./kyrgyz_spellcheck_model",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=10,
            save_steps=50,
            eval_steps=2,
            save_total_limit=3,
            fp16=True,
            eval_accumulation_steps=1024,
            use_wandb=False,
            run_name="kyrgyz-spellcheck-gemma"
        )
    )


    # Initialize trainer
    trainer = KyrgyzSpellCheckTrainer(config)

    # Run training
    trainer.train()

    # Example inference
    print("\n" + "="*50)
    print("TESTING INFERENCE")
    print("="*50)

    test_texts = [
        "Кыргызстан менин мекеним",
        "Биз достубуз болобуз",
        "Эртен мектепке барамын"
    ]

    for text in test_texts:
        corrected = trainer.inference(text)
        print(f"Original: {text}")
        print(f"Corrected: {corrected}")
        print("-" * 30)


if __name__ == "__main__":
    main()
