from enum import Enum
from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from .functional_impairment import Severity


class ScreeningStatus(str, Enum):
    NOT_INDICATED = "not_indicated"
    MILD_FEATURES_BELOW_THRESHOLD = "mild_features_below_diagnostic_threshold"
    ELEVATED_FEATURES = "elevated_features"
    SCREENING_RECOMMENDED = "screening_recommended"


class SleepDifficulty(str, Enum):
    NONE = "none"
    MILD_SLEEP_ONSET_DELAY = "mild_sleep_onset_delay"
    FREQUENT_NIGHT_WAKING = "frequent_night_waking"
    INSUFFICIENT_SLEEP = "insufficient_sleep"
    IRREGULAR_SCHEDULE = "irregular_schedule"


class SleepScreen(ContractModel):
    average_hours: Annotated[float, Field(ge=0, le=24)]
    difficulty: SleepDifficulty
    daytime_fatigue: Severity
    disorder: ScreeningStatus
