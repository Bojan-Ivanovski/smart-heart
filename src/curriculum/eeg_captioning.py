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
        source_sample = self.dataset[sample_index]
        analysis = source_sample["window_analysis"]
        if not isinstance(analysis, dict):
            raise ValueError(f"Invalid window analysis at sample {sample_index}.")
        answer = self._caption(analysis)
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
            source_samples=[source_sample],
        )

    @staticmethod
    def _caption(analysis: dict[str, object]) -> str:
        powers = analysis.get("relative_band_power")
        if not isinstance(powers, dict):
            raise ValueError("Window analysis has no relative band powers.")
        dominant = str(analysis["dominant_band"])
        spectral_description = ", ".join(
            f"{band} {float(powers[band]) * 100:.1f}%"
            for band in ("delta", "theta", "alpha", "beta", "gamma")
        )
        artifact_fraction = float(analysis["artifact_fraction"])
        if artifact_fraction < 0.01:
            quality = "minimal detected high-amplitude artifact"
        elif artifact_fraction < 0.05:
            quality = "a small detected high-amplitude artifact fraction"
        else:
            quality = "a notable detected high-amplitude artifact fraction"
        return (
            f"This window is dominated by {dominant}-band activity. "
            f"Relative band power is {spectral_description}. "
            "Temporal behavior is "
            f"{str(analysis['temporal_variability']).replace('_', ' ')}, with a "
            f"zero-crossing rate of {float(analysis['zero_crossing_rate']):.3f}. "
            f"Signal screening found {quality} ({artifact_fraction * 100:.1f}%)."
        )
