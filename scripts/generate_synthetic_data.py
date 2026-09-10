# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, rotate


ANGLES_DEG = np.arange(-75.0, 75.0 + 5.0, 5.0)


def add_poisson_gaussian_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    gaussian_sigma: float,
) -> np.ndarray:
    """Add Poisson counting noise and additive Gaussian noise."""
    noisy = rng.poisson(np.clip(image, 0, None)).astype(float)
    noisy += rng.normal(0.0, gaussian_sigma, image.shape)
    return np.clip(noisy, 0, None)


def ellipse_mask(shape, center, axes):
    """
    Create a filled ellipse mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape as (rows, cols).
    center : tuple[float, float]
        Ellipse center as (row, col).
    axes : tuple[float, float]
        Semi-axes as (radius_row, radius_col).
        If both are equal, this is a circle.
    """
    yy, xx = np.indices(shape)
    cy, cx = center
    ry, rx = axes

    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def simulate_eds_map(
    ideal_map: np.ndarray,
    rng: np.random.Generator,
    signal_counts: float = 15.0,
    background_counts: float = 1.0,
    psf_sigma: float = 2.5,
    gaussian_sigma: float = 0.01,
    post_blur_sigma: float = 0.5,
) -> np.ndarray:
    """Simulate a lower-SNR, lower-resolution EDS elemental map."""
    blurred_signal = gaussian_filter(
        ideal_map.astype(float),
        sigma=psf_sigma,
        mode="nearest",
    )

    expected_counts = background_counts + signal_counts * blurred_signal
    measured_counts = rng.poisson(expected_counts).astype(float)

    eds = (measured_counts - background_counts) / signal_counts
    eds += rng.normal(0.0, gaussian_sigma, ideal_map.shape)
    eds = np.clip(eds, 0, None)

    if post_blur_sigma > 0:
        eds = gaussian_filter(eds, sigma=post_blur_sigma, mode="nearest")

    return eds


def _forward_project(
    image: np.ndarray,
    angles_deg: np.ndarray,
) -> np.ndarray:
    """Generate a simple parallel-beam sinogram."""
    projections = []

    for angle in angles_deg:
        rotated = rotate(
            image,
            angle=angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        projections.append(rotated.sum(axis=0))

    return np.stack(projections, axis=1)


def _ramp_filter(sinogram: np.ndarray) -> np.ndarray:
    """Apply a ramp filter along the detector axis."""
    n_detector = sinogram.shape[0]
    frequencies = np.fft.rfftfreq(n_detector)
    ramp = 2.0 * np.abs(frequencies)

    spectrum = np.fft.rfft(sinogram, axis=0)
    return np.fft.irfft(
        spectrum * ramp[:, None],
        n=n_detector,
        axis=0,
    )


def _filtered_backprojection(
    sinogram: np.ndarray,
    angles_deg: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct a 2-D image by filtered backprojection."""
    filtered = _ramp_filter(sinogram)
    reconstruction = np.zeros(out_shape, dtype=float)

    for i, angle in enumerate(angles_deg):
        backprojection = np.tile(filtered[:, i], (out_shape[0], 1))
        reconstruction += rotate(
            backprojection,
            angle=-angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )

    reconstruction *= np.pi / (2.0 * len(angles_deg))
    return reconstruction


def _match_reference_scale(
    reconstruction: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Rescale a non-negative reconstruction to the reference intensity range."""
    reconstruction = np.clip(reconstruction, 0, None)

    recon_positive = reconstruction[reconstruction > 0]
    reference_positive = reference[reference > 0]

    if recon_positive.size == 0 or reference_positive.size == 0:
        return reconstruction

    recon_scale = np.percentile(recon_positive, 99.5)
    reference_scale = np.percentile(reference_positive, 99.5)

    if recon_scale > 0:
        reconstruction *= reference_scale / recon_scale

    return reconstruction


def simulate_fbp_reconstruction(
    ideal_map: np.ndarray,
    rng: np.random.Generator,
    signal_scale: float,
    background_counts: float,
    psf_sigma: float,
) -> np.ndarray:
    """
    Generate an FBP reconstruction from 31 limited-angle noisy projections.

    The angle range (-75 to +75 degrees in 5-degree increments) follows the
    experimental acquisition described in the original fusion paper. Limited
    angular coverage and projection noise create the characteristic
    star-shaped streak artifacts seen in FBP reconstructions.
    """
    source = gaussian_filter(
        ideal_map.astype(float),
        sigma=psf_sigma,
        mode="nearest",
    )

    sinogram = _forward_project(source, ANGLES_DEG)

    expected_counts = background_counts + signal_scale * np.clip(
        sinogram,
        0,
        None,
    )
    measured_counts = rng.poisson(expected_counts).astype(float)
    noisy_sinogram = (measured_counts - background_counts) / signal_scale

    reconstruction = _filtered_backprojection(
        noisy_sinogram,
        ANGLES_DEG,
        ideal_map.shape,
    )

    return _match_reference_scale(reconstruction, source)


def create_au_ag_nanoparticle(
    shape=(256, 256),
    center=None,
    particle_axes=(95, 95),
    au_axes=(55, 55),
    au_offset=(0, 0),
    seed=0,
):
    """
    Create a synthetic Au core / Ag shell nanoparticle.

    Returns
    -------
    haadf_reconstructions : dict[str, np.ndarray]
        Synthetic HAADF reconstructions keyed by algorithm name.
    au_reconstructions : dict[str, np.ndarray]
        Synthetic Au EDS reconstructions keyed by algorithm name.
    ag_reconstructions : dict[str, np.ndarray]
        Synthetic Ag EDS reconstructions keyed by algorithm name.
    au_gt : np.ndarray
        Clean binary Au ground truth.
    ag_gt : np.ndarray
        Clean binary Ag ground truth.
    """
    if center is None:
        center = (shape[0] // 2, shape[1] // 2)

    particle_center = center
    au_center = (
        center[0] + au_offset[0],
        center[1] + au_offset[1],
    )

    particle_mask = ellipse_mask(shape, particle_center, particle_axes)
    au_mask = ellipse_mask(shape, au_center, au_axes)

    # Keep the Au core inside the particle.
    au_mask = au_mask & particle_mask
    ag_mask = particle_mask & (~au_mask)

    # Clean elemental distributions used as ground truth.
    au_gt = au_mask.astype(float)
    ag_gt = ag_mask.astype(float)

    # Ideal HAADF signal: Au has stronger Z-contrast than Ag.
    haadf_ideal = np.zeros(shape, dtype=float)
    haadf_ideal[ag_mask] = 695.0
    haadf_ideal[au_mask] = 1086.0
    haadf_ideal = gaussian_filter(haadf_ideal, sigma=0.7, mode="nearest")

    # Independent, deterministic random streams for each modality/reconstruction.
    seed_sequence = np.random.SeedSequence(seed)
    rngs = [np.random.default_rng(s) for s in seed_sequence.spawn(6)]
    (
        rng_haadf_sirt,
        rng_haadf_fbp,
        rng_au_sirt,
        rng_au_fbp,
        rng_ag_sirt,
        rng_ag_fbp,
    ) = rngs

    # SIRT-like reconstructions: relatively smooth, with modality-specific noise.
    haadf_sirt = add_poisson_gaussian_noise(
        haadf_ideal,
        rng_haadf_sirt,
        gaussian_sigma=8.0,
    )

    au_sirt = simulate_eds_map(
        au_gt,
        rng_au_sirt,
        signal_counts=15.0,
        background_counts=1.0,
        psf_sigma=2.5,
        gaussian_sigma=0.01,
        post_blur_sigma=0.5,
    )

    ag_sirt = simulate_eds_map(
        ag_gt,
        rng_ag_sirt,
        signal_counts=15.0,
        background_counts=1.0,
        psf_sigma=2.5,
        gaussian_sigma=0.01,
        post_blur_sigma=0.5,
    )

    # FBP reconstructions: explicit limited-angle reconstruction to introduce
    # star-shaped streak artifacts similar to those in the original paper.
    haadf_fbp = simulate_fbp_reconstruction(
        haadf_ideal,
        rng_haadf_fbp,
        signal_scale=0.08,
        background_counts=2.0,
        psf_sigma=0.7,
    )

    au_fbp = simulate_fbp_reconstruction(
        au_gt,
        rng_au_fbp,
        signal_scale=0.5,
        background_counts=1.0,
        psf_sigma=1.5,
    )

    ag_fbp = simulate_fbp_reconstruction(
        ag_gt,
        rng_ag_fbp,
        signal_scale=0.5,
        background_counts=1.0,
        psf_sigma=1.5,
    )

    haadf_reconstructions = {"SIRT": haadf_sirt, "FBP": haadf_fbp}
    au_reconstructions = {"SIRT": au_sirt, "FBP": au_fbp}
    ag_reconstructions = {"SIRT": ag_sirt, "FBP": ag_fbp}

    return (
        haadf_reconstructions,
        au_reconstructions,
        ag_reconstructions,
        au_gt,
        ag_gt,
    )


def save_dataset(
    out_folder: Path,
    haadf: dict[str, np.ndarray],
    au: dict[str, np.ndarray],
    ag: dict[str, np.ndarray],
    au_gt: np.ndarray,
    ag_gt: np.ndarray,
) -> None:
    """Save reconstructions, ground truth, and manifest."""
    out_folder.mkdir(parents=True, exist_ok=True)
    gt_folder = out_folder / "ground_truth"
    gt_folder.mkdir(parents=True, exist_ok=True)

    np.savez(out_folder / "haadf.npz", **haadf)
    np.savez(out_folder / "Au.npz", **au)
    np.savez(out_folder / "Ag.npz", **ag)

    np.save(gt_folder / "Au_GT.npy", au_gt)
    np.save(gt_folder / "Ag_GT.npy", ag_gt)

    shape = list(next(iter(haadf.values())).shape)
    manifest = {
        "sample_id": "synthetic",
        "pixel_size_nm": None,
        "shape": shape,
        "modalities": {
            "Ag": {
                "file": "Ag.npz",
                "algorithms": ["SIRT", "FBP"],
            },
            "haadf": {
                "file": "haadf.npz",
                "algorithms": ["SIRT", "FBP"],
            },
            "Au": {
                "file": "Au.npz",
                "algorithms": ["SIRT", "FBP"],
            },
        },
    }

    with open(out_folder / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def plot_reconstructions(
    haadf: dict[str, np.ndarray],
    au: dict[str, np.ndarray],
    ag: dict[str, np.ndarray],
) -> None:
    """Preview SIRT and FBP reconstructions."""
    images = [
        haadf["SIRT"],
        au["SIRT"],
        ag["SIRT"],
        haadf["FBP"],
        au["FBP"],
        ag["FBP"],
    ]
    titles = [
        "HAADF SIRT",
        "Au SIRT",
        "Ag SIRT",
        "HAADF FBP",
        "Au FBP",
        "Ag FBP",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, image, title in zip(axes.ravel(), images, titles):
        vmax = np.percentile(image, 99.5)
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Generate and save the synthetic demo dataset."""
    (
        haadf,
        au,
        ag,
        au_gt,
        ag_gt,
    ) = create_au_ag_nanoparticle(
        particle_axes=(95, 90),
        au_axes=(55, 50),
        au_offset=(5, 20),
        seed=0,
    )

    # This script is intended to live in <repo>/scripts/.
    repo_root = Path(__file__).resolve().parents[1]
    out_folder = repo_root / "data" / "synthetic"

    save_dataset(
        out_folder,
        haadf,
        au,
        ag,
        au_gt,
        ag_gt,
    )
    plot_reconstructions(haadf, au, ag)


if __name__ == "__main__":
    main()
