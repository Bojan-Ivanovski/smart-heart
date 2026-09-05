from enum import Enum

from ..base import ContractModel
from ..types import NonEmptyText


class MedicationStatus(str, Enum):
    UNMEDICATED = "unmedicated"
    CURRENTLY_MEDICATED = "currently_medicated"
    MEDICATION_WITHHELD = "medication_withheld"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class Medication(ContractModel):
    adhd_medication_status: MedicationStatus
    medicated_during_recording: bool | None
    other_medications: list[NonEmptyText]
