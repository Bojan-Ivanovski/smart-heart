from ..base import ContractModel
from ..types import NonEmptyText


class AlternativeExplanation(ContractModel):
    factor: NonEmptyText
    assessment: NonEmptyText
