"""
Main trainer class for Kyrgyz spell checking with Gemma.
"""

import os
import json
import logging
import wandb
import torch
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
            bf16=False,
            dataloader_drop_last=False,
            eval_accumulation_steps=self.config.training.eval_accumulation_steps,
            report_to="wandb",
            run_name=self.config.training.run_name,
            # SFT specific parameters
            max_seq_length=self.config.data.max_length,
            completion_only_loss=True,
            packing=False,
        )

    def train(self):
        """Main training function."""
        logger.info("Starting training setup...")

        # Check if using Accelerate pipeline parallelism
        if (self.config.model.use_pipeline_parallelism and
            self.config.model.use_accelerate_pipeline):
            return self._train_with_accelerate_pipeline()
        else:
            return self._train_standard()

    def _train_with_accelerate_pipeline(self):
        """Training with Accelerate pipeline parallelism."""
        logger.info("Setting up training with Accelerate pipeline parallelism...")

        # Setup tokenizer first
        tokenizer = self.model_manager.setup_tokenizer()

        # Setup model with Accelerate pipeline (returns model and accelerator)
        model, accelerator = self.model_manager.setup_accelerate_pipeline_model()

        # Setup dataset manager and load data
        self.dataset_manager = DatasetManager(self.config, tokenizer)
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_manager.load_data()

        # Setup evaluator
        self.evaluator = EvaluationMetrics(tokenizer)

        # Initialize wandb if enabled
        if self.config.training.use_wandb:
            wandb.init(
                project="kyrgyz-spellcheck-accelerate",
                config=self.config.to_dict(),
                name=f"gemma-accelerate-pipeline-{self.config.training.num_train_epochs}epochs"
            )

        # Setup SFT configuration for Accelerate
        sft_config = self.setup_sft_config()

        # Modify config for Accelerate compatibility
        sft_config.dataloader_num_workers = 0  # Accelerate may need this
        sft_config.remove_unused_columns = False

        # Initialize SFTTrainer with Accelerate-prepared model
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            preprocess_logits_for_metrics=self.evaluator.preprocess_logits_for_metrics,
        )

        # The model is already prepared by Accelerate, so we can train directly
        trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        accelerator.wait_for_everyone()  # Ensure all processes are synchronized

        if accelerator.is_main_process:
            trainer.save_model()
            tokenizer.save_pretrained(self.config.training.output_dir)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        # Test evaluation
        logger.info("Running test evaluation...")
        test_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.evaluator.compute_metrics,
            preprocess_logits_for_metrics=self.evaluator.preprocess_logits_for_metrics,
        )
        test_results = test_trainer.evaluate()
        logger.info(f"Test results: {test_results}")

        # Save training config
        if accelerator.is_main_process:
            os.makedirs(self.config.training.output_dir, exist_ok=True)
            with open(os.path.join(self.config.training.output_dir, "training_config.json"), "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)

        if self.config.training.use_wandb and accelerator.is_main_process:
            wandb.log({"final_eval": eval_results, "test_results": test_results})
            wandb.finish()

        logger.info("Accelerate pipeline training completed successfully!")

    def _train_standard(self):
        """Standard training without Accelerate pipeline parallelism."""
        logger.info("Setting up standard training...")

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
                name=f"gemma-{self.config.training.num_train_epochs}epochs"
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

        # Train the model
        trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.training.output_dir)

        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")

        # Test evaluation
        logger.info("Running test evaluation...")
        test_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
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

        logger.info("Standard training completed successfully!")
