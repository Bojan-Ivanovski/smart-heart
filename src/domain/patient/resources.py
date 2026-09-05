from typing import Literal

from ..base import ContractModel


class PatientResources(ContractModel):
    clinical_profile: Literal["clinical_profile.json"]
    recordings: Literal["recordings.json"]
    segments: Literal["segments.json"]
    signals: Literal["signals.npz"]
