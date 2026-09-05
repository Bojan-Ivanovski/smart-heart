from ..data.loader import SmartHeartDataset

from .base import CurriculumDataset, OpenTSLMSample


class AttentionTaskCoT(CurriculumDataset):
    name = "stage3_attention_task_cot"
    scope = "recording"
    target_key = "attention_task_cot"

    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        super().__init__(dataset, eos_token)
        self._recordings = self._grouped_indices()

    def __len__(self) -> int:
        return len(self._recordings)

    def __getitem__(self, index: int) -> OpenTSLMSample:
        recording_indices = self._recordings[index]
        value = self._target(recording_indices[0])
        if not isinstance(value, dict) or not isinstance(value.get("target"), dict):
            raise ValueError(f"Invalid attention-task target at sample {index}.")

        references = {str(value) for value in value["input_references"]}
        referenced_indices = [
            sample_index
            for sample_index in recording_indices
            if self.dataset.sample_identity(sample_index)[2] in references
        ]
        representative_indices = self._representative_indices(referenced_indices)
        target = value["target"]
        return self._format_sample(
            representative_indices,
            pre_prompt=(
                "Analyze this attention-task EEG recording. First describe the "
                "observed signal evidence, then reason about the task-state "
                "pattern without treating EEG features as diagnostic proof."
            ),
            post_prompt=(
                "Provide a concise reasoning chain followed by a conclusion and "
                "confidence level."
            ),
            answer=self._answer(target),
        )

    @staticmethod
    def _answer(target: dict[str, object]) -> str:
        observations = "; ".join(str(value) for value in target["observations"])
        reasoning = " ".join(str(value) for value in target["reasoning_steps"])
        return (
            f"Observations: {observations} Reasoning: {reasoning} "
            f"Conclusion: {target['conclusion']} Confidence: {target['confidence']}."
        )
