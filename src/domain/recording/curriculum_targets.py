from ..base import ContractModel
from .reasoning_target import RecordingReasoningTarget


class RecordingCurriculumTargets(ContractModel):
    attention_task_cot: RecordingReasoningTarget | None = None
    resting_state_cot: RecordingReasoningTarget | None = None
