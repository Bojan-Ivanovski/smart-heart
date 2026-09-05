from enum import Enum
from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import Proportion
from .relative_band_power import FrequencyBand, RelativeBandPower


class TemporalVariability(str, Enum):
    STABLE = "stable"
    MODERATELY_VARIABLE = "moderately_variable"
    HIGHLY_VARIABLE = "highly_variable"


class AggregateFeatures(ContractModel):
    relative_band_power: RelativeBandPower
    dominant_band: FrequencyBand
    theta_beta_ratio: Annotated[float, Field(ge=0)]
    theta_alpha_ratio: Annotated[float, Field(ge=0)]
    spectral_entropy: Proportion
    channel_correlation_mean: Annotated[float, Field(ge=-1, le=1)]
    temporal_variability: TemporalVariability
