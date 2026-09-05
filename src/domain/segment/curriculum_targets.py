from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from .caption_target import CaptionTarget
from .multiple_choice_question import MultipleChoiceQuestion


class SegmentCurriculumTargets(ContractModel):
    mcq_warmup: Annotated[list[MultipleChoiceQuestion], Field(min_length=1)]
    eeg_captioning: CaptionTarget
