from ..base import ContractModel
from .diagnostic_target import DiagnosticTarget


class PatientCurriculumTargets(ContractModel):
    diagnostic_cot: DiagnosticTarget
