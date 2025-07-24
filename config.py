"""
Configuration classes for Gemma fine-tuning experiment.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    model_name: str = "google/gemma-3-1b-it"
    max_length: int = 256
    use_quantization: bool = False
    attn_implementation: str = "eager"
    use_peft: bool = True  # Flag to enable/disable PEFT training
    # Pipeline parallelism settings
    use_pipeline_parallelism: bool = False
    num_pipeline_stages: int = 4  # Number of GPUs for pipeline parallelism
    device_map_strategy: str = "custom"


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
