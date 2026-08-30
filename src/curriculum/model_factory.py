from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

from ..utility.runtime import Runtime


class OpenTSLMModelFactory:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def create(self, model_id: str, *, enable_lora: bool) -> OpenTSLMSP:
        model = OpenTSLMSP(
            llm_id=model_id,
            device=self.runtime.model_init_device,
        )
        if self.runtime.kind == "xla":
            model.to(self.runtime.device)
            model.device = self.runtime.device
        if enable_lora:
            model.enable_lora()
        return model
