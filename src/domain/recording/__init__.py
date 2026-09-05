from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..base import ContractModel
from ..types import Identifier, NonEmptyText
from .aggregate_features import AggregateFeatures, TemporalVariability
from .analysis import RecordingAnalysis
from .curriculum_targets import RecordingCurriculumTargets
from .preprocessing import Preprocessing
from .preprocessing_step import PreprocessingStep
from .quality import ArtifactSeverity, RecordingQuality
from .reasoning_output import Confidence, RecordingReasoningOutput
from .reasoning_target import RecordingReasoningTarget
from .relative_band_power import FrequencyBand, RelativeBandPower


class SignalRepresentation(str, Enum):
    RAW_EEG = "raw_eeg"
    BAND_POWER_TIMESERIES = "band_power_timeseries"


class Recording(ContractModel):
    recording_id: Identifier
    signal_path: Literal["signals.npz"]
    signal_key: Identifier
    shape: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]]
    dtype: Literal["float32"]
    representation: SignalRepresentation
    layout: Literal["channels_first"]
    sampling_rate_hz: Annotated[float, Field(gt=0)] | None
    channel_names: Annotated[list[NonEmptyText], Field(min_length=1)]
    condition: Identifier
    task: Identifier
    quality: RecordingQuality
    preprocessing: Preprocessing
    analysis: RecordingAnalysis
    curriculum_targets: RecordingCurriculumTargets

    @model_validator(mode="after")
    def validate_channel_count(self) -> Recording:
        if self.shape[0] != len(self.channel_names):
            raise ValueError("shape channel count must match channel_names")
        return self


from .document import RecordingsDocument

__all__ = [
    "AggregateFeatures",
    "ArtifactSeverity",
    "Confidence",
    "FrequencyBand",
    "Preprocessing",
    "PreprocessingStep",
    "Recording",
    "RecordingAnalysis",
    "RecordingCurriculumTargets",
    "RecordingQuality",
    "RecordingReasoningOutput",
    "RecordingReasoningTarget",
    "RecordingsDocument",
    "RelativeBandPower",
    "SignalRepresentation",
    "TemporalVariability",
]
