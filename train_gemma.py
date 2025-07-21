"""
Finetuning procedure for Gemma-3-1B-IT for Kyrgyz spell checking.
Uses LoRA (Low-Rank Adaptation) for efficient finetuning with SFTTrainer.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

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
from datasets import Dataset as HFDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "google/gemma-3-1b-it"
    max_length: int = 256
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
    max_length: int = 256
    max_val_samples: Optional[int] = None  # Manual limit for validation dataset size


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

        # Prepare formatted texts for SFT using apply_chat_template
        def format_data(misspelled_texts, correct_texts):
            formatted_texts = []
            for misspelled, correct in zip(misspelled_texts, correct_texts):
                instruction = "Correct the spelling mistakes in the following Kyrgyz text:"

                # Create conversation structure for chat template
                messages = [
                    {
                        "role": "user",
                        "content": f"{instruction}\n{misspelled}"
                    },
                    {
                        "role": "assistant",
                        "content": correct
                    }
                ]

                # Use apply_chat_template but remove the automatic BOS token
                # since SFTTrainer will add it during tokenization
                formatted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Remove the BOS token that apply_chat_template adds
                # since SFTTrainer will add its own BOS token
                if formatted_text.startswith(self.tokenizer.bos_token):
                    formatted_text = formatted_text[len(self.tokenizer.bos_token):]

                formatted_texts.append(formatted_text)

            return formatted_texts

        # Create Hugging Face datasets
        train_texts = format_data(splits['train'][0], splits['train'][1])
        val_texts = format_data(splits['val'][0], splits['val'][1])
        test_texts = format_data(splits['test'][0], splits['test'][1])

        # Limit validation dataset size if specified
        if self.config.data.max_val_samples is not None and len(val_texts) > self.config.data.max_val_samples:
            val_texts = val_texts[:self.config.data.max_val_samples]

        self.train_dataset = HFDataset.from_dict({"text": train_texts})
        self.val_dataset = HFDataset.from_dict({"text": val_texts})
        self.test_dataset = HFDataset.from_dict({"text": test_texts})

        logger.info(f"Datasets loaded:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Validation: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")

    def setup_sft_config(self):
        """Setup SFT configuration."""
        return SFTConfig(
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
            dataloader_drop_last=False,  # Add this to handle batch size consistency
            eval_accumulation_steps=self.config.training.eval_accumulation_steps,
            report_to="wandb" if self.config.training.use_wandb else None,
            run_name=self.config.training.run_name,
            # SFT specific parameters
            max_seq_length=self.config.data.max_length,
            completion_only_loss=True,
            packing=False,
        )

    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        print(predictions.shape, labels.shape, "predictions and labels shapes")

        # Handle predictions - they might be logits, so convert to token IDs
        if predictions.ndim > 2 or predictions.dtype != np.int64:
            # If predictions are logits, get the argmax
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            predictions = np.argmax(predictions, axis=-1)

        # Ensure predictions and labels are valid token IDs
        vocab_size = len(self.tokenizer)

        # Clip token IDs to valid range and convert to int
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int64)
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int64)

        # Replace any remaining invalid tokens with pad token
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        predictions = np.where(predictions < 0, pad_token_id, predictions)
        labels = np.where(labels < 0, pad_token_id, labels)

        try:
            # Decode predictions and labels
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Error decoding predictions/labels: {e}")
            # Return basic metrics if decoding fails
            return {
                "CER": 1.0,
                "WER": 1.0
            }

        overall_cer = jiwer.cer(decoded_labels, decoded_preds)
        overall_wer = jiwer.wer(decoded_labels, decoded_preds)

        return {
            "CER": overall_cer,
            "WER": overall_wer
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

        # Setup SFT configuration
        sft_config = self.setup_sft_config()

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
                temperature=0.0,
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
    # Set GPU visibility to use only first 2 GPUs
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    config = ExperimentConfig(
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
            num_samples=32,  # Increased samples for better training
            max_length=256
        ),
        training=TrainingConfig(
            output_dir="./kyrgyz_spellcheck_model",
            num_train_epochs=3,  # Increased epochs for better convergence
            per_device_train_batch_size=4,  # Increased batch size
            per_device_eval_batch_size=4,  # Larger eval batch size
            learning_rate=2e-7,  # Higher learning rate for LoRA
            weight_decay=0.01,
            warmup_steps=50,  # Reduced warmup steps (5% of total steps)
            logging_steps=25,  # Log less frequently
            save_steps=200,  # Save less frequently
            eval_steps=2,  # Evaluate less frequently
            save_total_limit=3,
            fp16=True,
            eval_accumulation_steps=32,  # Reduced for stability
            use_wandb=True,  # Enable wandb for better tracking
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
