from enum import Enum
from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from .symptom_domain import SymptomDomain


class SymptomPattern(str, Enum):
    PERSISTENT = "persistent"
    INTERMITTENT = "intermittent"
    SITUATIONAL = "situational"


class RecentChange(str, Enum):
    STABLE = "stable"
    IMPROVING = "improving"
    WORSENING = "worsening"


class Symptoms(ContractModel):
    duration_months: Annotated[int, Field(ge=0)]
    onset_age_years: Annotated[float, Field(ge=0, le=120)]
    inattention: SymptomDomain
    hyperactivity_impulsivity: SymptomDomain
    pattern: SymptomPattern
    recent_changes: RecentChange
