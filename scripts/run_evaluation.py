"""
Evaluate a saved fusion run against ground truth (synthetic data only).

Loads a run's saved fusion.npy + config.json, the corresponding denoised target,
and ground truth from data_dir/ground_truth/<target_modality>_GT.npy, then reports
three correlation coefficients:
    - fit agreement:  target   vs. fused        (already saved by run_fusion.py)
    - fused accuracy: ground truth vs. fused
    - target accuracy: ground truth vs. target  (baseline, pre-fusion)

Usage: edit scripts/eval_config.yaml, then run this script directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fusion_demo.io import load_dataset
from fusion_demo.denoise import denoise
from fusion_demo.evaluation import evaluate

CONFIG_PATH = Path(__file__).parent / "eval_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    config_dir = CONFIG_PATH.parent
    run_dir = (config_dir / eval_config["run_dir"]).resolve()

    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        run_config = json.load(f)

    data_dir = Path(run_config["data_dir"])
    target_modality = run_config["target_modality"]

    ground_truth_path = data_dir / "ground_truth" / f"{target_modality}_GT.npy"
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"No ground truth found for target modality '{target_modality}' "
            f"at {ground_truth_path}. Ground-truth evaluation only works "
            "for datasets that provide one."
        )
    ground_truth = np.load(ground_truth_path)

    fused = np.load(run_dir / "fusion.npy")
    dataset = load_dataset(data_dir, modality_names=[target_modality])
    target = denoise(dataset[target_modality])

    fit_agreement = evaluate(target, fused)
    fused_accuracy = evaluate(ground_truth, fused)
    target_accuracy = evaluate(ground_truth, target)

    print(f"Evaluating {run_dir} (target modality: '{target_modality}')")
    print(f"  fit agreement  (target vs. fused):        {fit_agreement}")
    print(f"  fused accuracy (ground truth vs. fused):   {fused_accuracy}")
    print(f"  target accuracy (ground truth vs. target): {target_accuracy}")


if __name__ == "__main__":
    main()
