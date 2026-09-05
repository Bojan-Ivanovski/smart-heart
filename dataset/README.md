# SmartHeart Dataset

This directory is the canonical, directly version-controlled SmartHeart dataset.
It is not generated at runtime and does not require archive extraction.

## Layout

```text
dataset/
  schema/
    clinical_profile.schema.json
    patient.schema.json
    recordings.schema.json
    segments.schema.json
  patients/
    <patient-id>/
      clinical_profile.json
      patient.json
      recordings.json
      signals.npz
      segments.json
```

`schema/` contains strict JSON Schema Draft 2020-12 contracts for every JSON
artifact in each generated patient directory. Generated patient data must pass the
corresponding schema plus cross-file reference and numerical-consistency validation.

Each `patient.json` contains patient-level metadata, diagnosis, recording
resources, and patient-level curriculum targets. Generated patient directories
separate clinical evidence, recording analysis, and segment-level targets into the
other JSON files. The loader discovers patients directly from
`patients/*/patient.json` and remains compatible with records awaiting generation.

Each `signals.npz` contains the patient's recordings as `float32`,
channels-first arrays. A recording's `signal_key` identifies its array. Runtime
model windows are runtime slices of these arrays and are not duplicated on disk.

## Contract Rules

- Patient IDs are globally unique and are the unit of dataset splitting.
- A patient and all their recordings always belong to one persisted split.
- Signals have shape `[channels, timesteps]` and use `float32` storage.
- Missing metadata is represented by `null`; it must not be invented.
- `segments` contains variable-length, signal-derived regions with meaningful targets.
- Each segment defines the default runtime window policy for model input.
- The CLI can override window size while preserving each segment policy's
  stride-to-size ratio and maximum-window limit.
- Runtime windowing stays inside segment boundaries and does not modify the recording.
- Every emitted window receives a cached spectral and time-domain analysis derived
  from its exact signal values.
- MCQ and EEG-caption supervision is derived from that window analysis. Recording
  and diagnostic reasoning continue to use their broader stored targets.
- Measured evidence, synthetic interpretation, and protected diagnosis remain separate.
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
