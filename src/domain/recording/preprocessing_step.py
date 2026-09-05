from typing import Any

from ..base import ContractModel
from ..types import Identifier


class PreprocessingStep(ContractModel):
    operation: Identifier
    parameters: dict[str, Any]
