from ..base import ContractModel
from ..types import EvidenceList
from .diagnostic_conclusion import DiagnosticConclusion


class DiagnosticOutput(ContractModel):
    eeg_evidence: EvidenceList
    clinical_evidence: EvidenceList
    alternative_explanations: EvidenceList
    reasoning_steps: EvidenceList
    conclusion: DiagnosticConclusion
