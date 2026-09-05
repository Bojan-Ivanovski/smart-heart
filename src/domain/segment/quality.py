from ..base import ContractModel
from ..types import Proportion


class SegmentQuality(ContractModel):
    usable: bool
    artifact_fraction: Proportion
