from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..curriculum import Curriculum
from ..curriculum.base import OpenTSLMSample
from ..data import SmartHeartDataset, build_dataloader
from ..models import ModelArchitecture, OpenTSLMModelFactory
from ..runtime import Runtime
from ..training import CheckpointManager


_GENERATION_TOKEN_LIMIT = 512


@dataclass(frozen=True)
class PredictionConfig:
    model_id: str = "google/gemma-3-1b-pt"
    architecture: ModelArchitecture | str = ModelArchitecture.SP
    enable_lora: bool = True
    gradient_checkpointing: bool = False
    checkpoint_root: Path = Path("checkpoints")

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", ModelArchitecture(self.architecture))
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")
        if self.architecture is ModelArchitecture.FLAMINGO and self.enable_lora:
            raise ValueError("LoRA is not supported for OpenTSLMFlamingo.")


@dataclass(frozen=True)
class PredictionResult:
    stage: str
    patient_id: str
    recording_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    pre_prompt: str
    post_prompt: str
    prediction: str
    checkpoint: Path


class Predictor:
    def __init__(
        self,
        runtime: Runtime,
        model_factory: OpenTSLMModelFactory,
    ) -> None:
        self.runtime = runtime
        self.model_factory = model_factory

    def predict(
        self,
        dataset: SmartHeartDataset,
        config: PredictionConfig,
        *,
        stage_name: str,
        index: int = 0,
        prompt: str | None = None,
    ) -> PredictionResult:
        if index < 0:
            raise ValueError("index must be at least 0.")
        if prompt is not None and not prompt.strip():
            raise ValueError("prompt cannot be empty when provided.")

        model = self.model_factory.create(
            config.model_id,
            architecture=config.architecture,
            enable_lora=config.enable_lora,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        eos_token = model.get_eos_token()
        if not eos_token:
            raise ValueError("The selected model tokenizer has no EOS token.")

        stage = Curriculum(dataset, str(eos_token)).get(stage_name)
        if not stage:
            raise ValueError(f"Stage '{stage_name}' has no samples.")
        if index >= len(stage):
            raise ValueError(
                f"index must be smaller than the stage size ({len(stage)})."
            )

        sample = stage[index].copy()
        if prompt is not None:
            sample["post_prompt"] = (
                f"{sample['post_prompt']}\nAdditional instruction: {prompt.strip()}"
            )

        checkpoint = CheckpointManager(
            config.checkpoint_root,
            config.model_id,
            config.architecture,
        ).require_latest()
        model.load_from_file(str(checkpoint))
        model.eval()

        batch = next(
            iter(
                build_dataloader(
                    _SingleSampleDataset(sample),
                    batch_size=1,
                    shuffle=False,
                )
            )
        )
        with torch.inference_mode():
            generated = model.generate(
                batch,
                max_new_tokens=_GENERATION_TOKEN_LIMIT,
            )
        if len(generated) != 1:
            raise ValueError(f"Expected one prediction, received {len(generated)}.")

        return PredictionResult(
            stage=stage.name,
            patient_id=str(sample["patient_id"]),
            recording_ids=tuple(str(value) for value in sample["recording_ids"]),
            segment_ids=tuple(str(value) for value in sample["segment_ids"]),
            pre_prompt=str(sample["pre_prompt"]),
            post_prompt=str(sample["post_prompt"]),
            prediction=self._without_eos(str(generated[0]), str(eos_token)),
            checkpoint=checkpoint,
        )

    @staticmethod
    def _without_eos(value: str, eos_token: str) -> str:
        if eos_token in value:
            value = value.split(eos_token, maxsplit=1)[0]
        return value.strip()


class _SingleSampleDataset(Dataset):
    def __init__(self, sample: OpenTSLMSample) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> OpenTSLMSample:
        if index != 0:
            raise IndexError(index)
        return self.sample
