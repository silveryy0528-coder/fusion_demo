"""
Run the cross-modality fusion pipeline end-to-end from a YAML config.

Usage: edit scripts/fusion_config.yaml, then run this script directly.
See scripts/fusion_config.yaml for the config schema.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from fusion_demo.pipeline import run_pipeline
from fusion_demo.io import load_dataset
from fusion_demo.denoise import denoise

from plotting import plot_fusion_summary

CONFIG_PATH = Path(__file__).parent / "fusion_config.yaml"


def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config_dir = CONFIG_PATH.parent
    data_dir = (config_dir / config["data_dir"]).resolve()
    results_dir = (config_dir / config["results_dir"]).resolve()

    target_modality = config["target_modality"]
    predictor_modalities = config["predictor_modalities"]
    filters = config["filters"]
    gauss_base = config.get("gauss_base", 2**0.5)
    gauss_depth = config.get("gauss_depth", 1)
    n_components = config.get("n_components")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir / run_name

    fused, _, metrics = run_pipeline(
        data_dir=data_dir,
        target_modality=target_modality,
        predictor_modalities=predictor_modalities,
        filters=filters,
        gauss_base=gauss_base,
        gauss_depth=gauss_depth,
        n_components=n_components,
        output_dir=output_dir,
    )
    print("\n----- Fusion run summary: -----")
    print(f"Run saved to {output_dir}")
    print(f"Fit agreement (target vs. fused): {metrics}")

    # Reload haadf + target for display in the summary figure.
    display_dataset = load_dataset(data_dir, modality_names=["haadf", target_modality])
    haadf = denoise(display_dataset["haadf"])
    target = denoise(display_dataset[target_modality])

    figure_path = output_dir / "summary.png"
    plot_fusion_summary(haadf, target, fused, save_path=figure_path)
    print(f"Summary figure saved to {figure_path}")


if __name__ == "__main__":
    main()
