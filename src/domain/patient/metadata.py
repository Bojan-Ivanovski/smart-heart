from enum import Enum
from typing import Annotated

from pydantic import StringConstraints

from ..base import ContractModel
from ..types import Identifier
from .annotation import Annotation
from .demographics import Demographics


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PatientMetadata(ContractModel):
    schema_version: Annotated[
        str,
        StringConstraints(pattern=r"^2\.0\.0(?:-draft)?$"),
    ]
    source_dataset: Identifier
    split_group: Identifier
    split: DatasetSplit
    annotation: Annotation
    demographics: Demographics
