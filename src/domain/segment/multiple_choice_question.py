from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from ..base import ContractModel
from ..types import Identifier, NonEmptyText
from .multiple_choice_options import MultipleChoiceOptions


class MultipleChoiceAnswer(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class MultipleChoiceQuestion(ContractModel):
    question_id: Identifier
    question: NonEmptyText
    options: MultipleChoiceOptions
    answer: MultipleChoiceAnswer
    answer_text: NonEmptyText
    explanation: NonEmptyText

    @model_validator(mode="after")
    def validate_answer_text(self) -> MultipleChoiceQuestion:
        option = getattr(self.options, self.answer.value.lower())
        if option != self.answer_text:
            raise ValueError("answer_text must match the selected option")
        return self
