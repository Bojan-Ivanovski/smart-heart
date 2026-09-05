from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import Identifier
from . import Segment


class SegmentsDocument(ContractModel):
    patient_id: Identifier
    segments: Annotated[list[Segment], Field(min_length=1)]
