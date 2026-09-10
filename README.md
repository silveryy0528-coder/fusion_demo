# fusion_demo

A self-contained refactor of a cross-modality (HAADF-STEM / EDS) tomographic
image fusion pipeline, built around Partial Least Squares (PLS) regression.

Given aligned orthoslices (or small volumes) from multiple imaging
modalities, it builds engineered feature images per modality and fits a PLS
model to predict a target element's (denoised) signal from the others —
useful when one modality is noisy/low-dose but correlates spatially with a
cleaner one.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Run the pipeline on the included synthetic dataset:

```bash
python scripts/run_fusion.py
```

This reads `scripts/fusion_config.yaml`, runs the fusion pipeline, and saves the
result, metrics, config, and a summary figure to `results/<timestamp>/`.
Edit `fusion_config.yaml` to point at a different dataset or change the target
modality / predictors / filters.

Run the test suite:

```bash
pytest
```

## Evaluating against ground truth (synthetic data only)

If a dataset provides ground truth (see `data/README.md`), you can evaluate
a previously saved run against it without re-running the fit:

```bash
python scripts/run_evaluation.py
```

Edit `scripts/eval_config.yaml` to point at a `results/<timestamp>/` folder.
It reports three correlation coefficients: fit agreement (target vs. fused,
already saved in that run's `metrics.json`), fused accuracy (ground truth
vs. fused), and target accuracy (ground truth vs. the noisy pre-fusion
target) -- the last two only possible because synthetic data comes with a
known ground truth.

## Structure

- `src/fusion_demo/` — the package: `io.py` (load/save), `denoise.py`
  (NMF-based target denoising), `features.py` (feature-image generation),
  `fusion.py` (PLS regression + model building), `evaluation.py` (metrics),
  `pipeline.py` (wires the above together).
- `scripts/` — `run_fusion.py` (entry point), `run_evaluation.py`
  (ground-truth evaluation of a saved run), and `plotting.py` (figures).
- `data/` — input datasets; see [`data/README.md`](data/README.md) for the format.
- `results/` — pipeline outputs (one timestamped folder per run).
- `tests/` — pytest suite, one file per module plus an end-to-end pipeline test.