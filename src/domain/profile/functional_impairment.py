from enum import Enum

from ..base import ContractModel


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class FunctionalSetting(str, Enum):
    SCHOOL = "school"
    HOME = "home"
    WORK = "work"
    SOCIAL = "social"


class FunctionalImpairment(ContractModel):
    settings: list[FunctionalSetting]
    academic: Severity
    social: Severity
    home: Severity
