from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )
