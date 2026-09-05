from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import Field, model_validator

from ..base import ContractModel


class WindowSelection(str, Enum):
    EVENLY_SPACED = "evenly_spaced"
    RANDOM = "random"


class WindowPolicy(ContractModel):
    size_samples: Annotated[int, Field(ge=1)]
    stride_samples: Annotated[int, Field(ge=1)]
    maximum_windows: Annotated[int, Field(ge=1)]
    selection: WindowSelection

    @model_validator(mode="after")
    def validate_stride(self) -> WindowPolicy:
        if self.stride_samples > self.size_samples:
            raise ValueError("window stride cannot exceed window size")
        return self
