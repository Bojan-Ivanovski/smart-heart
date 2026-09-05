from __future__ import annotations

from pydantic import model_validator

from ..base import ContractModel
from ..types import Identifier
from .annotation import Annotation
from .clinical_ground_truth import ClinicalGroundTruth, Diagnosis, LabelSource
from .curriculum_targets import PatientCurriculumTargets
from .demographics import AgeGroup, Demographics, Handedness, Sex
from .diagnostic_conclusion import (
    Confidence,
    DiagnosticConclusion,
    DiagnosticSeverity,
    Presentation,
)
from .diagnostic_input_references import DiagnosticInputReferences
from .diagnostic_output import DiagnosticOutput
from .diagnostic_target import DiagnosticTarget
from .metadata import DatasetSplit, PatientMetadata
from .resources import PatientResources


class Patient(ContractModel):
    patient_id: Identifier
    metadata: PatientMetadata
    resources: PatientResources
    clinical_ground_truth: ClinicalGroundTruth
    curriculum_targets: PatientCurriculumTargets

    @model_validator(mode="after")
    def validate_diagnostic_label(self) -> Patient:
        diagnostic_label = (
            self.curriculum_targets.diagnostic_cot.target.conclusion.diagnosis
        )
        if diagnostic_label != self.clinical_ground_truth.diagnosis:
            raise ValueError("diagnostic target must match clinical ground truth")
        return self

__all__ = [
    "AgeGroup",
    "Annotation",
    "ClinicalGroundTruth",
    "Confidence",
    "DatasetSplit",
    "Demographics",
    "Diagnosis",
    "DiagnosticConclusion",
    "DiagnosticInputReferences",
    "DiagnosticOutput",
    "DiagnosticSeverity",
    "DiagnosticTarget",
    "Handedness",
    "LabelSource",
    "Patient",
    "PatientCurriculumTargets",
    "PatientMetadata",
    "PatientResources",
    "Presentation",
    "Sex",
]
