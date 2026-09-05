from enum import Enum
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"


class DatasetSource(str, Enum):
    ALL = "all"
    CHILDREN = "adhd_children"
    COGNITIVE_FUNCTION = "adhd_cognitive_function"
    GAMEPLAY = "adhd_gameplay"

    @property
    def filter_value(self) -> str | None:
        return None if self is DatasetSource.ALL else self.value


class CurriculumStageName(str, Enum):
    MCQ = "stage1_mcq"
    EEG_CAPTIONING = "stage2_eeg_captioning"
    ATTENTION_TASK_COT = "stage3_attention_task_cot"
    RESTING_STATE_COT = "stage4_resting_state_cot"
    DIAGNOSTIC_COT = "stage5_diagnostic_cot"


class EvaluationSplit(str, Enum):
    VALIDATION = "validation"
    TEST = "test"


def stage_names(stages: list[CurriculumStageName] | None) -> tuple[str, ...] | None:
    if not stages:
        return None
    return tuple(stage.value for stage in stages)
