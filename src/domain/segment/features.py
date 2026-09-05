from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..recording.relative_band_power import FrequencyBand, RelativeBandPower
from ..types import Proportion


class SegmentFeatures(ContractModel):
    relative_band_power: RelativeBandPower
    dominant_band: FrequencyBand
    theta_beta_ratio: Annotated[float, Field(ge=0)]
    spectral_entropy: Proportion
    rms_amplitude_normalized: Annotated[float, Field(ge=0)]
    zero_crossing_rate: Proportion
