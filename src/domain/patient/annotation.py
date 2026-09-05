from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import NonEmptyText


class Annotation(ContractModel):
    seed: Annotated[int, Field(ge=0)]
    generation_basis: Annotated[list[NonEmptyText], Field(min_length=1)]
