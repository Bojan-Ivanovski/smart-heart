from __future__ import annotations

from enum import Enum

from pydantic import model_validator

from ..base import ContractModel
from ..types import Proportion


class FrequencyBand(str, Enum):
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"


class RelativeBandPower(ContractModel):
    delta: Proportion
    theta: Proportion
    alpha: Proportion
    beta: Proportion
    gamma: Proportion

    @model_validator(mode="after")
    def validate_total_power(self) -> RelativeBandPower:
        if abs(sum(self.model_dump().values()) - 1.0) > 1e-5:
            raise ValueError("relative band power must sum to 1")
        return self
