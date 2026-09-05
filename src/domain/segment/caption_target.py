from typing import Literal

from ..base import ContractModel
from ..types import Identifier
from .caption_output import CaptionOutput


class CaptionTarget(ContractModel):
    scope: Literal["segment"]
    supervision: Identifier
    target: CaptionOutput
