from enum import Enum

from ..base import ContractModel


class Diagnosis(str, Enum):
    ADHD = "ADHD"
    NOT_ADHD = "NOT_ADHD"
    UNCERTAIN = "UNCERTAIN"


class LabelSource(str, Enum):
    SOURCE_DATASET_GROUP = "source_dataset_group"
    CLINICAL_ASSESSMENT = "clinical_assessment"
    RESEARCH_PROTOCOL = "research_protocol"
    SYNTHETIC_GENERATION = "synthetic_generation"
    UNKNOWN = "unknown"


class ClinicalGroundTruth(ContractModel):
    diagnosis: Diagnosis
    label_source: LabelSource
    verified: bool
