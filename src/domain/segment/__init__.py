from __future__ import annotations

from typing import Annotated

from pydantic import Field, model_validator

from ..base import ContractModel
from ..types import Identifier, NonEmptyText
from .caption_output import CaptionOutput
from .caption_target import CaptionTarget
from .curriculum_targets import SegmentCurriculumTargets
from .features import SegmentFeatures
from .multiple_choice_options import MultipleChoiceOptions
from .multiple_choice_question import MultipleChoiceAnswer, MultipleChoiceQuestion
from .quality import SegmentQuality
from .window_policy import WindowPolicy, WindowSelection
from .window_analysis import WindowAnalysis


class Segment(ContractModel):
    segment_id: Identifier
    recording_id: Identifier
    start_timestep: Annotated[int, Field(ge=0)]
    end_timestep: Annotated[int, Field(ge=1)]
    duration_seconds: Annotated[float, Field(gt=0)] | None
    segment_type: Identifier
    selection_reason: NonEmptyText
    quality: SegmentQuality
    measured_features: SegmentFeatures
    window_policy: WindowPolicy
    curriculum_targets: SegmentCurriculumTargets

    @model_validator(mode="after")
    def validate_bounds(self) -> Segment:
        if self.end_timestep <= self.start_timestep:
            raise ValueError("segment end must be after segment start")
        return self


from .document import SegmentsDocument

__all__ = [
    "CaptionOutput",
    "CaptionTarget",
    "MultipleChoiceAnswer",
    "MultipleChoiceOptions",
    "MultipleChoiceQuestion",
    "Segment",
    "SegmentCurriculumTargets",
    "SegmentFeatures",
    "SegmentQuality",
    "SegmentsDocument",
    "WindowPolicy",
    "WindowAnalysis",
    "WindowSelection",
]
