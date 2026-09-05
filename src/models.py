from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from .runtime import Runtime, RuntimeKind

if TYPE_CHECKING:
    from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

    OpenTSLMModel: TypeAlias = OpenTSLMSP | OpenTSLMFlamingo
else:
    OpenTSLMModel: TypeAlias = object


class ModelArchitecture(str, Enum):
    SP = "sp"
    FLAMINGO = "flamingo"


class OpenTSLMModelFactory:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def create(
        self,
        model_id: str,
        *,
        architecture: ModelArchitecture | str = ModelArchitecture.SP,
        enable_lora: bool = True,
        gradient_checkpointing: bool = False,
    ) -> OpenTSLMModel:
        try:
            model_architecture = ModelArchitecture(architecture)
        except ValueError as error:
            supported = ", ".join(value.value for value in ModelArchitecture)
            raise ValueError(
                f"Unknown model architecture '{architecture}'. "
                f"Expected one of: {supported}."
            ) from error

        if model_architecture is ModelArchitecture.SP:
            model = self._create_sp(model_id, enable_lora)
        else:
            if enable_lora:
                raise ValueError(
                    "LoRA is supported by OpenTSLMSP, not OpenTSLMFlamingo."
                )
            model = self._create_flamingo(model_id, gradient_checkpointing)

        if self.runtime.kind is RuntimeKind.XLA:
            model.to(self.runtime.device)
            model.device = self.runtime.device
        return model

    def _create_sp(self, model_id: str, enable_lora: bool) -> OpenTSLMModel:
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

        model = OpenTSLMSP(
            llm_id=model_id,
            device=self.runtime.model_init_device,
        )
        if enable_lora:
            model.enable_lora()
        return model

    def _create_flamingo(
        self,
        model_id: str,
        gradient_checkpointing: bool,
    ) -> OpenTSLMModel:
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

        return OpenTSLMFlamingo(
            llm_id=model_id,
            device=self.runtime.model_init_device,
            cross_attn_every_n_layers=1,
            gradient_checkpointing=gradient_checkpointing,
        )
