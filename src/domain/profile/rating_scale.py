from enum import Enum
from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import NonEmptyText


class RatingInterpretation(str, Enum):
    LOW = "low"
    BORDERLINE = "borderline"
    ELEVATED = "elevated"


class RatingScale(ContractModel):
    instrument: NonEmptyText
    score: Annotated[float, Field(ge=0)]
    maximum: Annotated[float, Field(gt=0)]
    interpretation: RatingInterpretation
