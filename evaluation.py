"""
Evaluation metrics and utilities for Kyrgyz spell checking.
"""

import logging
import numpy as np
import jiwer

# Add wandb import (optional)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Handles computation of evaluation metrics."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred

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

        metrics = {
            "val_CER": overall_cer,
            "val_WER": overall_wer
        }

        # Report to wandb if available and initialized
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics)

        return metrics

    @staticmethod
    def preprocess_logits_for_metrics(logits, labels):
        """Preprocess logits for metrics computation."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
