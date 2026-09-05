from ..data.loader import SmartHeartDataset

from .base import CurriculumDataset, OpenTSLMSample


class MCQ(CurriculumDataset):
    name = "stage1_mcq"
    scope = "segment"
    target_key = "mcq_warmup"

    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        super().__init__(dataset, eos_token)
        self._questions: list[tuple[int, int]] = []
        for index in self._matching_indices():
            questions = self._target(index)
            if not isinstance(questions, list) or len(questions) != 2:
                raise ValueError(f"Invalid MCQ target at sample {index}.")
            self._questions.extend(
                (index, question_index)
                for question_index in range(2)
            )

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, index: int) -> OpenTSLMSample:
        sample_index, question_index = self._questions[index]
        source_sample = self.dataset[sample_index]
        analysis = source_sample["window_analysis"]
        if not isinstance(analysis, dict):
            raise ValueError(f"Invalid window analysis at sample {sample_index}.")
        question = self._window_question(
            source_sample,
            analysis,
            question_index,
        )
        options = question["options"]
        option_text = "\n".join(
            f"{letter}. {options[letter]}" for letter in ("A", "B", "C", "D")
        )
        result = self._format_sample(
            [sample_index],
            pre_prompt=(
                "Analyze the EEG time series and answer the multiple-choice "
                "question using the measured signal pattern."
            ),
            post_prompt=(
                f"Question: {question['question']}\n{option_text}\n"
                "Answer with only A, B, C, or D."
            ),
            answer=str(question["answer"]),
            source_samples=[source_sample],
        )
        result["question_id"] = str(question["question_id"])
        return result

    @staticmethod
    def _window_question(
        sample: dict[str, object],
        analysis: dict[str, object],
        question_index: int,
    ) -> dict[str, object]:
        powers = analysis.get("relative_band_power")
        if not isinstance(powers, dict):
            raise ValueError("Window analysis has no relative band powers.")
        prefix = (
            f"{sample['segment_id']}__window-{sample['window_start']}-"
            f"{sample['window_end']}"
        )
        if question_index == 0:
            bands = ("delta", "theta", "alpha", "beta", "gamma")
            ranked = sorted(
                bands,
                key=lambda band: (-float(powers[band]), bands.index(band)),
            )
            included = set(ranked[:4])
            option_bands = [band for band in bands if band in included]
            options = {
                letter: band.title()
                for letter, band in zip(("A", "B", "C", "D"), option_bands)
            }
            answer = ("A", "B", "C", "D")[option_bands.index(ranked[0])]
            return {
                "question_id": f"{prefix}__mcq-001",
                "question": (
                    "Among the listed options, which frequency band has the "
                    "greatest relative power in this window?"
                ),
                "options": options,
                "answer": answer,
            }

        theta = float(powers["theta"])
        beta = float(powers["beta"])
        if abs(theta - beta) <= 1e-6:
            answer = "C"
        else:
            answer = "A" if theta > beta else "B"
        return {
            "question_id": f"{prefix}__mcq-002",
            "question": "How do theta and beta relative power compare in this window?",
            "options": {
                "A": "Theta power is higher",
                "B": "Beta power is higher",
                "C": "They are equal",
                "D": "The comparison cannot be determined",
            },
            "answer": answer,
        }
