from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import Identifier
from . import Recording


class RecordingsDocument(ContractModel):
    patient_id: Identifier
    recordings: Annotated[list[Recording], Field(min_length=1)]
