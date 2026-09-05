from enum import Enum

from ..base import ContractModel
from ..types import EvidenceList, NonEmptyText


class Confidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RecordingReasoningOutput(ContractModel):
    observations: EvidenceList
    reasoning_steps: EvidenceList
    conclusion: NonEmptyText
    confidence: Confidence
