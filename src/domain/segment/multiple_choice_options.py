from pydantic import Field

from ..base import ContractModel
from ..types import NonEmptyText


class MultipleChoiceOptions(ContractModel):
    a: NonEmptyText = Field(alias="A")
    b: NonEmptyText = Field(alias="B")
    c: NonEmptyText = Field(alias="C")
    d: NonEmptyText = Field(alias="D")
