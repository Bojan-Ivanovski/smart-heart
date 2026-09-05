from dataclasses import dataclass
from pathlib import Path

from ..models import ModelArchitecture


@dataclass(frozen=True)
class EvaluationConfig:
    model_id: str = "google/gemma-3-270m"
    architecture: ModelArchitecture | str = ModelArchitecture.SP
    batch_size: int = 1
    enable_lora: bool = True
    gradient_checkpointing: bool = False
    checkpoint_root: Path = Path("checkpoints")
    output_root: Path = Path("results")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "architecture",
            ModelArchitecture(self.architecture),
        )
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.architecture is ModelArchitecture.FLAMINGO and self.enable_lora:
            raise ValueError("LoRA is not supported for OpenTSLMFlamingo.")
