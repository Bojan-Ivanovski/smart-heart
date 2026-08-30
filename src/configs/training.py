from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str = "google/gemma-3-270m"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    enable_lora: bool = True
    checkpoint_root: Path = Path("checkpoints")
    fresh_start: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
