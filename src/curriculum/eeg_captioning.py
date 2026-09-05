from ..data.loader import SmartHeartDataset

from .base import CurriculumDataset, OpenTSLMSample


class EEGCaptioning(CurriculumDataset):
    name = "stage2_eeg_captioning"
    scope = "segment"
    target_key = "eeg_captioning"

    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        super().__init__(dataset, eos_token)
        self._indices = self._matching_indices()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> OpenTSLMSample:
        sample_index = self._indices[index]
        value = self._target(sample_index)
        if not isinstance(value, dict) or not isinstance(value.get("target"), dict):
            raise ValueError(f"Invalid EEG caption target at sample {sample_index}.")
        target = value["target"]
        answer = " ".join(
            (
                str(target["summary"]),
                str(target["spectral_description"]),
                str(target["temporal_description"]),
                str(target["quality_note"]),
            )
        )
        return self._format_sample(
            [sample_index],
            pre_prompt=(
                "Describe the EEG time series objectively. Focus on signal "
                "quality, spectral balance, and temporal behavior."
            ),
            post_prompt=(
                "Write one concise clinical-style caption. Do not infer a "
                "diagnosis from this segment."
            ),
            answer=answer,
        )
