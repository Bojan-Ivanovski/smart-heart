from enum import Enum

from ..base import ContractModel
from ..types import NonEmptyText


class ReferralSource(str, Enum):
    PARENT = "parent"
    SCHOOL_TEACHER = "school_teacher"
    SCHOOL_COUNSELOR = "school_counselor"
    SELF = "self"
    CLINICIAN = "clinician"
    RESEARCH_PROTOCOL = "research_protocol"
    OTHER = "other"


class Referral(ContractModel):
    reason: NonEmptyText
    referral_source: ReferralSource
