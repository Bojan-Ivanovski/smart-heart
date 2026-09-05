from enum import Enum
from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import NonEmptyText


class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    OLDER_ADULT = "older_adult"


class Handedness(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    AMBIDEXTROUS = "ambidextrous"
    UNKNOWN = "unknown"


class Demographics(ContractModel):
    sex: Sex
    age_years: Annotated[float, Field(ge=0, le=120)]
    age_group: AgeGroup
    handedness: Handedness
    education_context: NonEmptyText
