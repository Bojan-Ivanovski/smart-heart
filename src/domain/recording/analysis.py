from typing import Annotated

from pydantic import Field

from ..base import ContractModel
from ..types import Identifier
from .aggregate_features import AggregateFeatures


class RecordingAnalysis(ContractModel):
    method: Identifier
    feature_version: Annotated[str, Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")]
    aggregate_features: AggregateFeatures
    interpretation_flags: list[Identifier]
