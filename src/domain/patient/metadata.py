from typing import Annotated

from pydantic import StringConstraints

from ..base import ContractModel
from ..types import Identifier
from .annotation import Annotation
from .demographics import Demographics


class PatientMetadata(ContractModel):
    schema_version: Annotated[
        str,
        StringConstraints(pattern=r"^2\.0\.0(?:-draft)?$"),
    ]
    source_dataset: Identifier
    annotation: Annotation
    demographics: Demographics
