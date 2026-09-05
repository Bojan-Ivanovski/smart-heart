from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..domain.patient import Patient
from ..domain.profile import ClinicalProfile
from ..domain.recording import Recording, RecordingsDocument
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

    recording_by_id = _index_recordings(recordings.recordings, patient.patient_id)
    segment_ids: set[str] = set()
    segment_ids_by_recording: dict[str, set[str]] = {
        recording_id: set() for recording_id in recording_by_id
    }
    for segment in segments.segments:
        if segment.segment_id in segment_ids:
            raise ValueError(f"Duplicate segment ID '{segment.segment_id}'.")
        segment_ids.add(segment.segment_id)
        recording = recording_by_id.get(segment.recording_id)
        if recording is None:
            raise ValueError(
                f"Segment '{segment.segment_id}' references unknown recording "
                f"'{segment.recording_id}'."
            )
        if segment.end_timestep > recording.shape[1]:
            raise ValueError(
                f"Segment '{segment.segment_id}' exceeds recording "
                f"'{recording.recording_id}'."
            )
        segment_ids_by_recording[recording.recording_id].add(segment.segment_id)

    for recording in recordings.recordings:
        valid_segment_ids = segment_ids_by_recording[recording.recording_id]
        for target in (
            recording.curriculum_targets.attention_task_cot,
            recording.curriculum_targets.resting_state_cot,
        ):
            if target is None:
                continue
            unknown_ids = set(target.input_references) - valid_segment_ids
            if unknown_ids:
                raise ValueError(
                    f"Recording '{recording.recording_id}' target references "
                    f"unknown segments: {sorted(unknown_ids)}."
                )

    diagnostic_recordings = set(
        patient.curriculum_targets.diagnostic_cot.input_references.recording_ids
    )
    unknown_recordings = diagnostic_recordings - recording_by_id.keys()
    if unknown_recordings:
        raise ValueError(
            "Diagnostic target references unknown recordings: "
            f"{sorted(unknown_recordings)}."
        )

    return patient, profile, recordings, segments


def _index_recordings(
    recordings: list[Recording],
    patient_id: str,
) -> dict[str, Recording]:
    recording_by_id: dict[str, Recording] = {}
    for recording in recordings:
        if recording.recording_id in recording_by_id:
            raise ValueError(
                f"Duplicate recording ID '{recording.recording_id}' "
                f"for patient '{patient_id}'."
            )
        recording_by_id[recording.recording_id] = recording
    return recording_by_id
