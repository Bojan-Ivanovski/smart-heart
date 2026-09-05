from ..base import ContractModel
from ..types import NonEmptyText


class CaptionOutput(ContractModel):
    summary: NonEmptyText
    spectral_description: NonEmptyText
    temporal_description: NonEmptyText
    quality_note: NonEmptyText
