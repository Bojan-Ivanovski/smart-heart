# SmartHeart Dataset

This directory is the canonical, directly version-controlled SmartHeart dataset.
It is not generated at runtime and does not require archive extraction.

## Layout

```text
dataset/
  patients/
    <patient-id>/
      patient.json
      signals.npz
```

Each `patient.json` contains patient-level metadata, diagnosis, recording
metadata, optional explicitly annotated windows, and curriculum targets.
The loader discovers these files directly from `patients/*/patient.json`.

Each `signals.npz` contains the patient's recordings as `float32`,
channels-first arrays. A recording's `signal_key` identifies its array. Runtime
windows are slices of these arrays and are not duplicated on disk.

## Contract Rules

- Patient IDs are globally unique and are the unit of dataset splitting.
- A patient and all their recordings always belong to one persisted split.
- Signals have shape `[channels, timesteps]` and use `float32` storage.
- Missing metadata is represented by `null`; it must not be invented.
- `windows` is reserved for explicitly annotated intervals and can be empty.
- Runtime windowing does not modify the stored recording.
- Curriculum evidence must identify its supervision source.
- `diagnostic_cot.target.diagnosis` is currently the only universal target.

## Sources

- `adhd_children`: 121 children recorded during a visual-attention task using
  19 EEG channels at 128 Hz.
- `adhd_cognitive_function`: 79 valid adults across resting, cognitive, and
  auditory conditions using paired EEG channels at 256 Hz. One publisher-marked
  corrupted subject was excluded during normalization.
- `adhd_gameplay`: 10 gameplay subjects with five-band power time series. The
  source does not provide a reliable sampling rate for these derived values.

The source datasets have been normalized into this contract. The canonical
files in this directory are the training source of truth.
