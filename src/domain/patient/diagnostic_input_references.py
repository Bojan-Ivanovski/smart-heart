from typing import Annotated, Literal

from pydantic import Field

from ..base import ContractModel
from ..types import Identifier


class DiagnosticInputReferences(ContractModel):
    recording_ids: Annotated[list[Identifier], Field(min_length=1)]
    clinical_profile: Literal["clinical_profile.json"]
