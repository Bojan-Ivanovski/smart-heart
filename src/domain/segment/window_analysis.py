from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..recording.aggregate_features import TemporalVariability
from ..recording.relative_band_power import FrequencyBand, RelativeBandPower
from ..types import Proportion


class WindowAnalysis(ContractModel):
    relative_band_power: RelativeBandPower
    dominant_band: FrequencyBand
    theta_beta_ratio: Annotated[float, Field(ge=0)]
    theta_alpha_ratio: Annotated[float, Field(ge=0)]
    spectral_entropy: Proportion
    channel_correlation_mean: Annotated[float, Field(ge=-1, le=1)]
    temporal_variability: TemporalVariability
    artifact_fraction: Proportion
    rms_amplitude_normalized: Annotated[float, Field(ge=0)]
    zero_crossing_rate: Proportion
