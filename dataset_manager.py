"""
Data processing and dataset management for Kyrgyz spell checking.
"""

import logging
from datasets import Dataset as HFDataset
from data import KyrgyzDataLoader
from config import ExperimentConfig

logger = logging.getLogger(__name__)


class DatasetManager:
    """Handles dataset loading and preprocessing for training."""
    
    def __init__(self, config: ExperimentConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
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
        train_texts = self._format_data(splits['train'][0], splits['train'][1])
        val_texts = self._format_data(splits['val'][0], splits['val'][1])
        test_texts = self._format_data(splits['test'][0], splits['test'][1])

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
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _format_data(self, misspelled_texts, correct_texts):
        """Format data for SFT using apply_chat_template."""
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
