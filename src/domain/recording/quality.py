from enum import Enum

from ..base import ContractModel
from ..types import Identifier, Proportion


class ArtifactSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class RecordingQuality(ContractModel):
    usable: bool
    artifact: ArtifactSeverity | None
    usable_fraction: Proportion
    artifact_types: list[Identifier]
