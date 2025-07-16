"""
Data loader module for Kyrgyz spelling correction dataset.
Handles loading data from HuggingFace datasets and various file formats.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KyrgyzSpellCheckDataset:
    """Dataset class for Kyrgyz spelling correction."""

    def __init__(self, misspelled_texts: List[str], correct_texts: List[str]):
        """
        Initialize dataset with provided lists.

        Args:
            misspelled_texts: List of misspelled sentences
            correct_texts: List of corresponding correct sentences
        """
        assert len(misspelled_texts) == len(correct_texts), "Lists must have equal length"
        self.misspelled = misspelled_texts
        self.correct = correct_texts
        logger.info(f"Initialized dataset with {len(self.misspelled)} sentence pairs")

    @classmethod
    def from_hf_dataset(cls, dataset_path: str, num_samples: int = None, **kwargs):
        """
        Create dataset from HuggingFace dataset using KyrgyzDataLoader.

        Args:
            dataset_path: Path to HuggingFace dataset directory
            num_samples: Number of samples to select from each split
            **kwargs: Additional arguments for data loader

        Returns:
            Dictionary with train/val/test KyrgyzSpellCheckDataset instances
        """
        loader = KyrgyzDataLoader()
        splits = loader.load_from_hf_dataset(dataset_path, num_samples=num_samples, **kwargs)

        return {
            'train': cls(splits['train'][0], splits['train'][1]),
            'val': cls(splits['val'][0], splits['val'][1]),
            'test': cls(splits['test'][0], splits['test'][1])
        }

class KyrgyzDataLoader:
    """
    A simplified data loader class for Kyrgyz spelling correction datasets.
    Primarily works with HuggingFace datasets but also supports file formats.
    """

    def __init__(self):
        """Initialize the data loader."""
        self.supported_formats = ['.csv', '.json', '.txt', '.tsv']

    def load_from_hf_dataset(self, dataset_path: str, num_samples: int = None,
                            misspelled_col: str = 'trash', correct_col: str = 'clean') -> Dict[str, Tuple[List[str], List[str]]]:
        """
        Load data from HuggingFace dataset saved to disk.

        Args:
            dataset_path: Path to the saved HuggingFace dataset
            num_samples: Number of samples to select from each split (None for all)
            misspelled_col: Column name for misspelled text
            correct_col: Column name for correct text

        Returns:
            Dictionary with train/val/test splits as tuples of (misspelled, correct) lists
        """
        try:
            # Load the dataset from disk
            dataset = load_from_disk(dataset_path)
            logger.info(f"Loaded dataset from {dataset_path}")

            # Create train/val split from the original train set
            train_val_dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)

            # Select samples if specified
            if num_samples:
                train_dataset = train_val_dataset['train'].select(range(min(num_samples, len(train_val_dataset['train']))))
                val_dataset = train_val_dataset['test'].select(range(min(num_samples, len(train_val_dataset['test']))))
                test_dataset = dataset['test'].select(range(min(num_samples, len(dataset['test']))))
            else:
                train_dataset = train_val_dataset['train']
                val_dataset = train_val_dataset['test']
                test_dataset = dataset['test']

            # Extract text pairs
            def extract_texts(ds):
                misspelled = ds[misspelled_col]
                correct = ds[correct_col]
                return misspelled, correct

            train_miss, train_corr = extract_texts(train_dataset)
            val_miss, val_corr = extract_texts(val_dataset)
            test_miss, test_corr = extract_texts(test_dataset)

            splits = {
                'train': (train_miss, train_corr),
                'val': (val_miss, val_corr),
                'test': (test_miss, test_corr)
            }

            logger.info(f"Dataset splits created:")
            logger.info(f"  Train: {len(train_miss)} pairs")
            logger.info(f"  Validation: {len(val_miss)} pairs")
            logger.info(f"  Test: {len(test_miss)} pairs")

            return splits

        except Exception as e:
            logger.error(f"Error loading HuggingFace dataset from {dataset_path}: {str(e)}")
            raise


    def validate_data(self, misspelled: List[str], correct: List[str]) -> bool:
        """Validate loaded data for quality and consistency."""
        if len(misspelled) != len(correct):
            logger.warning(f"Mismatched lengths: {len(misspelled)} vs {len(correct)}")
            return False

        if len(misspelled) == 0:
            logger.warning("No data loaded")
            return False

        # Check for empty strings
        empty_misspelled = sum(1 for text in misspelled if not text.strip())
        empty_correct = sum(1 for text in correct if not text.strip())

        if empty_misspelled > 0 or empty_correct > 0:
            logger.warning(f"Found {empty_misspelled} empty misspelled texts and "
                          f"{empty_correct} empty correct texts")

        # Basic statistics
        avg_misspelled_len = sum(len(text) for text in misspelled) / len(misspelled)
        avg_correct_len = sum(len(text) for text in correct) / len(correct)

        logger.info(f"Data validation summary:")
        logger.info(f"  Total pairs: {len(misspelled)}")
        logger.info(f"  Average misspelled length: {avg_misspelled_len:.1f} characters")
        logger.info(f"  Average correct length: {avg_correct_len:.1f} characters")

        return True


# Simplified utility function
def create_sample_data_file(output_path: str = "sample_kyrgyz_data.csv") -> None:
    """Create a sample data file for testing."""
    sample_data = {
        'misspelled': [
            "Кыргызстан менин мекеним",
            "Биз достубуз болобуз",
            "Эртен мектепке барамын",
            "Кыргыз тили сонун тил",
            "Мен китеп окуп жатам"
        ],
        'correct': [
            "Кыргызстан менин мекеним",
            "Биз достор болобуз",
            "Эртең мектепке барамын",
            "Кыргыз тили сонун тил",
            "Мен китеп окуп жатам"
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Sample data file created: {output_path}")


if __name__ == "__main__":
    # Example usage
    loader = KyrgyzDataLoader()

    # Example 1: Load from HuggingFace dataset (if available)
    splits = loader.load_from_hf_dataset('../../KSC/misspelled_kg_dataset/', num_samples=None)
    print("Simplified data loader demonstration completed successfully!")

    loader.validate_data(splits['train'][0], splits['train'][1])


