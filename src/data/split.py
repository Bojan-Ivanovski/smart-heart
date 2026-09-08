from collections.abc import Sequence
from enum import Enum

from sklearn.model_selection import train_test_split

from ..domain.patient import Patient


DEFAULT_SPLIT_SEED = 42


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def assign_patient_splits(
    patients: Sequence[Patient],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, DatasetSplit]:
    """Create a deterministic 60/20/20 split stratified by cohort and diagnosis."""
    if len(patients) < 3:
        raise ValueError(
            "At least three patients are required to create dataset splits."
        )

    ordered = sorted(patients, key=lambda patient: patient.patient_id)
    patient_ids = [patient.patient_id for patient in ordered]
    strata = [
        f"{patient.metadata.source_dataset}:{patient.clinical_ground_truth.diagnosis.value}"
        for patient in ordered
    ]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("Patient IDs must be globally unique before splitting.")

    train_ids, held_out_ids = train_test_split(
        patient_ids,
        train_size=0.6,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    stratum_by_id = dict(zip(patient_ids, strata))
    validation_ids, test_ids = train_test_split(
        held_out_ids,
        train_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=[stratum_by_id[patient_id] for patient_id in held_out_ids],
    )

    assignments = {
        patient_id: DatasetSplit.TRAIN for patient_id in train_ids
    }
    assignments.update(
        {patient_id: DatasetSplit.VALIDATION for patient_id in validation_ids}
    )
    assignments.update(
        {patient_id: DatasetSplit.TEST for patient_id in test_ids}
    )
    return assignments
