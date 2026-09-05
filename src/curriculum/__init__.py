from collections.abc import Iterator
from typing import TypeAlias

from ..data.loader import SmartHeartDataset

from .attention_task_cot import AttentionTaskCoT
from .diagnostic_cot import DiagnosticCoT
from .eeg_captioning import EEGCaptioning
from .mcq import MCQ
from .resting_state_cot import RestingStateCoT


CurriculumStage: TypeAlias = (
    MCQ
    | EEGCaptioning
    | AttentionTaskCoT
    | RestingStateCoT
    | DiagnosticCoT
)


class Curriculum:
    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        self.mcq = MCQ(dataset, eos_token)
        self.eeg_captioning = EEGCaptioning(dataset, eos_token)
        self.attention_task_cot = AttentionTaskCoT(dataset, eos_token)
        self.resting_state_cot = RestingStateCoT(dataset, eos_token)
        self.diagnostic_cot = DiagnosticCoT(dataset, eos_token)

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        return (
            self.mcq,
            self.eeg_captioning,
            self.attention_task_cot,
            self.resting_state_cot,
            self.diagnostic_cot,
        )

    def __iter__(self) -> Iterator[CurriculumStage]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def get(self, name: str) -> CurriculumStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"Unknown curriculum stage '{name}'.")


__all__ = [
    "AttentionTaskCoT",
    "Curriculum",
    "CurriculumStage",
    "DiagnosticCoT",
    "EEGCaptioning",
    "MCQ",
    "RestingStateCoT",
]
