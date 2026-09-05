from typing import Annotated, Literal

from pydantic import Field

from ..base import ContractModel
from ..types import NonEmptyText


class SymptomDomain(ContractModel):
    count: Annotated[int, Field(ge=0, le=9)]
    maximum: Literal[9]
    examples: Annotated[list[NonEmptyText], Field(max_length=9)]
