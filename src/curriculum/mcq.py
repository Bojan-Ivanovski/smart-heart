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
            if not isinstance(questions, list) or not questions:
                raise ValueError(f"Invalid MCQ target at sample {index}.")
            self._questions.extend(
                (index, question_index)
                for question_index in range(len(questions))
            )

    def __len__(self) -> int:
        return len(self._questions)

    def __getitem__(self, index: int) -> OpenTSLMSample:
        sample_index, question_index = self._questions[index]
        questions = self._target(sample_index)
        if not isinstance(questions, list):
            raise ValueError(f"Invalid MCQ target at sample {sample_index}.")
        question = questions[question_index]
        if not isinstance(question, dict):
            raise ValueError(f"Invalid MCQ question at sample {sample_index}.")

        options = question.get("options")
        if not isinstance(options, dict):
            raise ValueError("An MCQ question must contain an options object.")
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
        )
        result["question_id"] = str(question["question_id"])
        return result
