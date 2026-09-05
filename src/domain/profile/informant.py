from enum import Enum

from ..base import ContractModel
from ..types import NonEmptyText


class InformantRole(str, Enum):
    PARENT = "parent"
    TEACHER = "teacher"
    CAREGIVER = "caregiver"
    SELF = "self"
    CLINICIAN = "clinician"


class Informant(ContractModel):
    role: InformantRole
    summary: NonEmptyText
