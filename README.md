# SmartHeart

## Project Goal

The goal of SmartHeart is to train a model using OpenTSLM technology so it can understand time-series data. In this project, the focus is on brain-signal time-series data, enabling the model to classify patterns and support reasoning about why a given individual might have ADHD.

## Current Status

The repository now includes an early implementation focused on preparing EEG-based ADHD data for OpenTSLM. The current prototype loads patient `.mat` files from the `adhd_children_dataset`, converts them into per-patient samples, and builds a PyTorch `DataLoader` that prepares batched time-series inputs using OpenTSLM utilities.

## Environment

The project currently targets Python `3.12`, as defined in `.python-version`. Python dependencies are listed in `requirements.txt`, including PyTorch, SciPy, Hugging Face tooling, and OpenTSLM.

## Datasets

More information about the datasets, including the current dataset map and naming conventions, can be found in [datasets/README.md](datasets/README.md).

## Source Code

More information about the current implementation can be found in [src/README.md](src/README.md).

## Repository Layout

- `README.md`: project overview and workspace notes.
- `.gitignore`: keeps local datasets, Python cache files, and temporary artifacts out of version control.
- `.python-version`: defines the Python version used for local development.
- `requirements.txt`: lists the current Python dependencies for the project.
- `datasets/`: not committed; all datasets are stored here.
- `src/`: source code and implementation files.
- `tmp/`: temporary workspace files used during local experimentation.
