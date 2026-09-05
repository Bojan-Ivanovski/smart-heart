from typing import Annotated, Literal

from pydantic import Field

from ..base import ContractModel
from ..types import Identifier
from .reasoning_output import RecordingReasoningOutput


class RecordingReasoningTarget(ContractModel):
    scope: Literal["recording"]
    supervision: Identifier
    input_references: Annotated[list[Identifier], Field(min_length=1)]
    target: RecordingReasoningOutput
