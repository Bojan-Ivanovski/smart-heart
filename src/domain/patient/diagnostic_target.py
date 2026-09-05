from typing import Literal

from ..base import ContractModel
from ..types import NonEmptyText
from .diagnostic_input_references import DiagnosticInputReferences
from .diagnostic_output import DiagnosticOutput


class DiagnosticTarget(ContractModel):
    scope: Literal["patient"]
    supervision: NonEmptyText
    input_references: DiagnosticInputReferences
    target: DiagnosticOutput
