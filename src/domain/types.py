from typing import Annotated

from pydantic import Field, StringConstraints


Identifier = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$"),
]
NonEmptyText = Annotated[str, StringConstraints(min_length=1)]
EvidenceList = Annotated[list[NonEmptyText], Field(min_length=1)]
Proportion = Annotated[float, Field(ge=0, le=1)]
