from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    model_id: str = "google/gemma-3-270m"
    enable_lora: bool = True
    checkpoint_root: Path = Path("checkpoints")
    max_new_tokens: int = 2
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be at least 1 when provided")
