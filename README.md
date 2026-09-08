# SmartHeart

## Project Goal

The goal of SmartHeart is to train a model using OpenTSLM technology so it can understand time-series data. In this project, the focus is on brain-signal time-series data, enabling the model to classify patterns and support reasoning about why a given individual might have ADHD.

## Current Status

The repository includes a canonical patient-level EEG dataset and an OpenTSLM training pipeline. It supports raw EEG and gameplay band-power recordings, configurable time-series windows, deterministic patient-level dataset splits, LoRA training, deterministic seeds, automatic checkpoint reuse, and automatic XLA, CUDA, MPS, or CPU selection.

## Environment

The project currently targets Python `3.12`, as defined in `.python-version`. Base CPU and CUDA dependencies are listed in `requirements.txt`. TPU environments should install `requirements-tpu.txt`, which adds PyTorch/XLA and the Cloud TPU runtime.

## Datasets

The canonical dataset contract, layout, and source summary are documented in [dataset/README.md](dataset/README.md).
The model-facing preview, training, evaluation, and prediction commands use the
`adhd_cognitive_function` cohort. The other canonical cohorts remain available
for dataset analysis and future experiments.

## Source Code

The implementation is organized under `src/` into CLI, curriculum, data,
domain, evaluation, and training layers.

## Repository Layout

- `README.md`: project overview and workspace notes.
- `.gitignore`: keeps Python cache files, local environment files, and temporary artifacts out of version control.
- `.python-version`: defines the Python version used for local development.
- `requirements.txt`: lists the current Python dependencies for the project.
- `dataset/`: version-controlled canonical metadata and compressed patient signals.
- `src/`: package containing the CLI, data pipeline, runtime selection, model factory, and trainer.
- `tmp/`: temporary workspace files used during local experimentation.

## Usage

Preview the first MCQ model input and expected answer without initializing a
language model:

```powershell
python -m src.main preview
```

Generate one prediction from the final diagnostic checkpoint without running
evaluation metrics. The command prints structured input, prompt, prediction, and
expected-answer panels. It selects a random test sample unless `--index` is
provided:

```powershell
python -m src.main predict
```

Start training explicitly:

```powershell
python -m src.main train --device auto
```

Evaluate the latest saved checkpoint on every curriculum task in the validation
partition:

```powershell
python -m src.main evaluate --split validation --device auto
```

Training automatically loads the highest-numbered checkpoint for the model and
architecture. Each training command then creates the next chronological snapshot,
such as `checkpoint_2_stage_1.pt`; the suffix records which curriculum stage ran,
not the model's total curriculum coverage. Validation and prediction also load the
highest-numbered checkpoint. To discard the checkpoint history and train from the
base model, add `--fresh-start`. Checkpoints contain model weights, so reuse is a
warm start rather than an optimizer or epoch resume.

Run `python -m src.main --help` and a command's `--help` option for the complete CLI.
