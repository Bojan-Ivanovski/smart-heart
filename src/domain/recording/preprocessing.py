from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from .preprocessing_step import PreprocessingStep


class Preprocessing(ContractModel):
    source_timesteps: Annotated[int, Field(ge=1)]
    stored_timesteps: Annotated[int, Field(ge=1)]
    steps: list[PreprocessingStep]
