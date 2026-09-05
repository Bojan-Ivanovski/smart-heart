from ..data.loader import SmartHeartDataset

from .base import CurriculumDataset, OpenTSLMSample


class DiagnosticCoT(CurriculumDataset):
    name = "stage5_diagnostic_cot"
    scope = "patient"
    target_key = "diagnostic_cot"

    def __init__(self, dataset: SmartHeartDataset, eos_token: str) -> None:
        super().__init__(dataset, eos_token)
        self._patients = self._grouped_indices()

    def __len__(self) -> int:
        return len(self._patients)

    def __getitem__(self, index: int) -> OpenTSLMSample:
        patient_indices = self._patients[index]
        value = self._target(patient_indices[0])
        if not isinstance(value, dict) or not isinstance(value.get("target"), dict):
            raise ValueError(f"Invalid diagnostic target at sample {index}.")

        input_references = value.get("input_references")
        if not isinstance(input_references, dict):
            raise ValueError("A diagnostic target requires input references.")
        recording_ids = {
            str(recording_id)
            for recording_id in input_references["recording_ids"]
        }
        referenced_indices = [
            sample_index
            for sample_index in patient_indices
            if self.dataset.sample_identity(sample_index)[1] in recording_ids
        ]
        representative_indices = self._representative_indices(referenced_indices)
        profile = self.dataset.clinical_profile(representative_indices[0])

        return self._format_sample(
            representative_indices,
            pre_prompt=(
                "Integrate the patient's clinical history with the supplied EEG "
                "recordings. EEG is supporting evidence only and must not be "
                "treated as independently diagnostic."
            ),
            post_prompt=(
                f"Clinical profile: {self._profile_summary(profile)}\n"
                "Explain the EEG evidence, clinical evidence, and reasonable "
                "alternatives, then give the diagnosis, presentation, severity, "
                "and confidence."
            ),
            answer=self._answer(value["target"]),
        )

    @staticmethod
    def _profile_summary(profile: dict[str, object]) -> str:
        referral = profile["referral"]
        symptoms = profile["symptoms"]
        impairment = profile["functional_impairment"]
        informants = profile["informants"]
        rating_scales = profile["rating_scales"]
        alternatives = profile["alternative_explanations"]
        if not (
            isinstance(referral, dict)
            and isinstance(symptoms, dict)
            and isinstance(impairment, dict)
            and isinstance(informants, list)
            and isinstance(rating_scales, list)
            and isinstance(alternatives, list)
        ):
            raise ValueError("The clinical profile has an invalid structure.")

        inattention = symptoms["inattention"]
        hyperactivity = symptoms["hyperactivity_impulsivity"]
        if not isinstance(inattention, dict) or not isinstance(hyperactivity, dict):
            raise ValueError("The clinical symptom profile has an invalid structure.")
        informant_text = " ".join(
            f"{value['role']}: {value['summary']}"
            for value in informants
            if isinstance(value, dict)
        )
        scale_text = "; ".join(
            f"{value['instrument']} {value['score']}/{value['maximum']} "
            f"({value['interpretation']})"
            for value in rating_scales
            if isinstance(value, dict)
        )
        alternative_text = "; ".join(
            f"{value['factor']}: {value['assessment']}"
            for value in alternatives
            if isinstance(value, dict)
        ) or "none recorded"
        return (
            f"Referral: {referral['reason']} Symptoms have lasted "
            f"{symptoms['duration_months']} months with {inattention['count']} "
            f"inattentive and {hyperactivity['count']} hyperactive/impulsive "
            f"features; pattern: {symptoms['pattern']}. Functional settings: "
            f"{', '.join(str(value) for value in impairment['settings'])}. "
            f"Informants: {informant_text} Rating scales: {scale_text}. "
            f"Alternative explanations: {alternative_text}."
        )

    @staticmethod
    def _answer(target: dict[str, object]) -> str:
        eeg = "; ".join(str(value) for value in target["eeg_evidence"])
        clinical = "; ".join(str(value) for value in target["clinical_evidence"])
        alternatives = "; ".join(
            str(value) for value in target["alternative_explanations"]
        )
        reasoning = " ".join(str(value) for value in target["reasoning_steps"])
        conclusion = target["conclusion"]
        if not isinstance(conclusion, dict):
            raise ValueError("A diagnostic conclusion must be an object.")
        return (
            f"EEG evidence: {eeg} Clinical evidence: {clinical} "
            f"Alternative explanations: {alternatives} Reasoning: {reasoning} "
            f"Conclusion: diagnosis {conclusion['diagnosis']}, presentation "
            f"{conclusion['presentation']}, severity {conclusion['severity']}, "
            f"confidence {conclusion['confidence']}."
        )
