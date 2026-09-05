from enum import Enum

from ..base import ContractModel
from .clinical_ground_truth import Diagnosis


class Presentation(str, Enum):
    PREDOMINANTLY_INATTENTIVE = "predominantly_inattentive"
    PREDOMINANTLY_HYPERACTIVE_IMPULSIVE = "predominantly_hyperactive_impulsive"
    COMBINED = "combined"
    UNSPECIFIED = "unspecified"
    NOT_APPLICABLE = "not_applicable"


class DiagnosticSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    NOT_APPLICABLE = "not_applicable"


class Confidence(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class DiagnosticConclusion(ContractModel):
    diagnosis: Diagnosis
    presentation: Presentation
    severity: DiagnosticSeverity
    confidence: Confidence
