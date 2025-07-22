"""
Main trainer class for Kyrgyz spell checking with Gemma.
"""

import os
import json
import logging
import wandb
from trl import SFTTrainer, SFTConfig
from config import ExperimentConfig
from model_manager import ModelManager
from dataset_manager import DatasetManager
from evaluation import EvaluationMetrics

logger = logging.getLogger(__name__)


class KyrgyzSpellCheckTrainer:
    """Main trainer class for Kyrgyz spell checking model."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.dataset_manager = None
        self.evaluator = None

        # Components to be initialized
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
            dataloader_drop_last=False,
            eval_accumulation_steps=self.config.training.eval_accumulation_steps,
            report_to="wandb" if self.config.training.use_wandb else None,
            run_name=self.config.training.run_name,
            # SFT specific parameters
            max_seq_length=self.config.data.max_length,
            completion_only_loss=True,
            packing=False,
        )

    def train(self):
        """Main training function."""
        logger.info("Starting training setup...")

        # Setup components
        tokenizer = self.model_manager.setup_tokenizer()
        model = self.model_manager.setup_model()

        # Setup dataset manager and load data
        self.dataset_manager = DatasetManager(self.config, tokenizer)
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.load_data()

        # Setup evaluator
        self.evaluator = EvaluationMetrics(tokenizer)

        # Initialize wandb if enabled
        if self.config.training.use_wandb:
            wandb.init(
                project="kyrgyz-spellcheck",
                config=self.config.to_dict(),
                name=f"gemma-kyrgyz-spellcheck-{self.config.training.num_train_epochs}epochs"
            )

        # Setup SFT configuration
        sft_config = self.setup_sft_config()

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            preprocess_logits_for_metrics=self.evaluator.preprocess_logits_for_metrics,
        )

        # Start training (uncomment to enable actual training)
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.training.output_dir)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        # Test evaluation - create a separate trainer for test dataset
        logger.info("Running test evaluation...")
        test_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,  # Required but not used for evaluation
            eval_dataset=self.test_dataset,    # Use test dataset as eval dataset
            compute_metrics=self.evaluator.compute_metrics,
            preprocess_logits_for_metrics=self.evaluator.preprocess_logits_for_metrics,
        )
        test_results = test_trainer.evaluate()
        logger.info(f"Test results: {test_results}")

        # Save training config
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        with open(os.path.join(self.config.training.output_dir, "training_config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        if self.config.training.use_wandb:
            wandb.log({"final_eval": eval_results, "test_results": test_results})
            wandb.finish()

        logger.info("Training completed successfully!")

    def inference(self, text: str, max_new_tokens: int = 200) -> str:
        """Run inference on a single text."""
        return self.model_manager.inference(text, max_new_tokens)
