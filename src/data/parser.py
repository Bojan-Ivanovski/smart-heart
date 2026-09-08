from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..domain.patient import Patient
from ..domain.profile import ClinicalProfile
from ..domain.recording import RecordingsDocument
from ..domain.segment import SegmentsDocument


ModelType = TypeVar("ModelType", bound=BaseModel)
PatientDocuments = tuple[
    Patient,
    ClinicalProfile,
    RecordingsDocument,
    SegmentsDocument,
]


def parse_json(path: Path, model: type[ModelType]) -> ModelType:
    try:
        content = path.read_text(encoding="utf-8")
        return model.model_validate_json(content)
    except OSError as error:
        raise ValueError(f"Could not read '{path}': {error}") from error
    except ValidationError as error:
        raise ValueError(f"Invalid {model.__name__} in '{path}':\n{error}") from error


def parse_patient(directory: Path) -> PatientDocuments:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Patient directory does not exist: '{directory}'.")

    patient = parse_json(directory / "patient.json", Patient)
    if patient.patient_id != directory.name:
        raise ValueError(
            f"Patient ID '{patient.patient_id}' does not match directory "
            f"'{directory.name}'."
        )

    profile = parse_json(
        directory / patient.resources.clinical_profile,
        ClinicalProfile,
    )
    recordings = parse_json(
        directory / patient.resources.recordings,
        RecordingsDocument,
    )
    segments = parse_json(
        directory / patient.resources.segments,
        SegmentsDocument,
    )

    if recordings.patient_id != patient.patient_id:
        raise ValueError(f"Recording patient ID mismatch for '{patient.patient_id}'.")
    if segments.patient_id != patient.patient_id:
        raise ValueError(f"Segment patient ID mismatch for '{patient.patient_id}'.")

    return patient, profile, recordings, segments
