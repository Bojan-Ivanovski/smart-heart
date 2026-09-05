from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from .alternative_explanation import AlternativeExplanation
from .comorbidity_screen import ComorbidityScreen
from .functional_impairment import FunctionalImpairment, FunctionalSetting, Severity
from .informant import Informant, InformantRole
from .medication import Medication, MedicationStatus
from .rating_scale import RatingInterpretation, RatingScale
from .referral import Referral, ReferralSource
from .sleep_screen import ScreeningStatus, SleepDifficulty, SleepScreen
from .symptom_domain import SymptomDomain
from .symptoms import RecentChange, SymptomPattern, Symptoms


class ClinicalProfile(ContractModel):
    referral: Referral
    comorbidity_screen: ComorbidityScreen
    alternative_explanations: list[AlternativeExplanation]
    informants: Annotated[list[Informant], Field(min_length=1)]
    functional_impairment: FunctionalImpairment
    symptoms: Symptoms
    rating_scales: Annotated[list[RatingScale], Field(min_length=1)]
    medication: Medication

__all__ = [
    "AlternativeExplanation",
    "ClinicalProfile",
    "ComorbidityScreen",
    "FunctionalImpairment",
    "FunctionalSetting",
    "Informant",
    "InformantRole",
    "Medication",
    "MedicationStatus",
    "RatingInterpretation",
    "RatingScale",
    "RecentChange",
    "Referral",
    "ReferralSource",
    "ScreeningStatus",
    "Severity",
    "SleepDifficulty",
    "SleepScreen",
    "SymptomDomain",
    "SymptomPattern",
    "Symptoms",
]
