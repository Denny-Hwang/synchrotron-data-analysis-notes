# Deep Learning Hallucination

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting |
| **Noise Type** | Computational |
| **Severity** | Critical |
| **Frequency** | Occasional |
| **Detection Difficulty** | Hard |

## Visual Examples

```
Ground truth          DL-denoised output        Difference (hallucination)

┌───────────────┐    ┌───────────────┐          ┌───────────────┐
│               │    │     ●         │          │     ●         │
│   Noisy but   │    │   Clean +     │          │  FALSE feature │
│   featureless │ →  │   hallucinated│    =     │  that does NOT │
│   region      │    │   particle    │          │  exist in data │
│               │    │               │          │               │
└───────────────┘    └───────────────┘          └───────────────┘

  Low-SNR input        Network "helpfully"        Residual reveals
                       invents structure           the fabrication
```

> **⚠️ CRITICAL WARNING:** Deep learning reconstructions and denoising outputs must NEVER be
> trusted as ground truth for quantitative scientific analysis without independent validation.
> Networks can generate plausible but entirely fictitious features, especially in low-SNR
> regions or when processing data outside the training distribution. Always compare DL outputs
> with conventional reconstructions and perform uncertainty quantification before drawing
> scientific conclusions.

> **External references:**
> - [Gottschling et al. — The troublesome kernel (hallucinations in DL imaging)](https://doi.org/10.1137/20M1387237)
> - [Antun et al. — Instabilities of deep learning in image reconstruction](https://doi.org/10.1073/pnas.1907377117)

## Description

Deep learning hallucinations are features that appear in neural network-processed images but do not correspond to real structures in the sample. These artifacts arise because neural networks are generative models that learn statistical priors from training data — when presented with ambiguous or noisy input, they fill in plausible details that match the learned distribution rather than the actual physical reality. Hallucinations are particularly dangerous because they appear natural and self-consistent, making them nearly impossible to distinguish from real features without independent verification.

## Root Cause

Neural networks minimize a loss function over training data, learning both true signal structure and statistical biases present in the training set. When the network encounters input that differs from its training distribution (out-of-distribution data, unusual sample morphologies, different noise levels, or different acquisition parameters), it extrapolates using learned priors rather than the measured signal. Denoising networks can hallucinate textures and particles in featureless regions; super-resolution networks can invent sub-resolution structures; and reconstruction networks can create coherent but false features in undersampled regions. The problem is exacerbated by overconfident networks that produce high-certainty outputs regardless of input quality, giving no indication that the output is unreliable.

## Quick Diagnosis

```python
import numpy as np

# Compare DL output with conventional reconstruction
# dl_recon = ...     # DL-processed image
# conv_recon = ...   # conventional reconstruction (e.g., FBP, gridrec)
residual = dl_recon - conv_recon
# Hallucinations appear as structured features in the residual
residual_snr = np.std(residual) / np.mean(np.abs(conv_recon) + 1e-10)
print(f"Residual relative magnitude: {residual_snr:.4f}")
print(f"If residual contains structured features (not just noise), hallucination likely")
```

## Detection Methods

### Visual Indicators

- Features in the DL output that are absent from conventional reconstruction of the same data.
- Suspiciously clean or artifact-free results from very noisy or undersampled input data.
- Repetitive or template-like structures that resemble training data motifs.
- Features that change substantially with small perturbations of the input (adversarial instability).
- Resolution-limited features that appear sharper than the physical resolution of the instrument.

### Automated Detection

```python
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr


def detect_dl_hallucination(dl_output, conventional_output,
                             noise_level_estimate=None,
                             structure_threshold=0.15,
                             num_perturbations=5, perturbation_sigma=0.02):
    """
    Detect potential deep learning hallucinations by comparing DL output
    with conventional reconstruction and testing output stability.

    Parameters
    ----------
    dl_output : np.ndarray
        2D image from DL reconstruction/denoising.
    conventional_output : np.ndarray
        2D image from conventional method (FBP, gridrec, etc.).
    noise_level_estimate : float or None
        Estimated noise std of the conventional output. If None, estimated
        from high-frequency content.
    structure_threshold : float
        Threshold for structured residual detection (0 to 1).
    num_perturbations : int
        Number of input perturbations for stability testing.
    perturbation_sigma : float
        Relative amplitude of perturbation noise.

    Returns
    -------
    dict with keys:
        'residual_structure_score' : float — 0 (noise-like) to 1 (structured)
        'suspicious_regions' : np.ndarray — binary mask of suspect areas
        'stability_score' : float — 1.0 (stable) to 0.0 (unstable)
        'has_hallucination_risk' : bool
        'diagnostics' : str
    """
    # 1. Residual analysis
    # Normalize both to same scale
    dl_norm = (dl_output - np.mean(dl_output)) / (np.std(dl_output) + 1e-10)
    conv_norm = (conventional_output - np.mean(conventional_output)) / (
        np.std(conventional_output) + 1e-10
    )
    residual = dl_norm - conv_norm

    # Estimate noise level from conventional output if not provided
    if noise_level_estimate is None:
        # Use high-pass filtered image to estimate noise floor
        smoothed = gaussian_filter(conventional_output, sigma=3)
        noise_level_estimate = np.std(conventional_output - smoothed)

    # 2. Detect structured content in residual using autocorrelation
    residual_centered = residual - np.mean(residual)
    # Compute local variance as structure indicator
    local_var = gaussian_filter(residual_centered**2, sigma=10)
    global_var = np.var(residual_centered)

    # Structure score: ratio of max local variance to global variance
    # Pure noise has uniform variance; structured residual has hot spots
    structure_score = float(np.max(local_var) / (global_var + 1e-10))
    structure_score = min(structure_score / 5.0, 1.0)  # normalize to [0, 1]

    # 3. Identify suspicious regions (high residual that looks structured)
    residual_smooth = gaussian_filter(np.abs(residual_centered), sigma=5)
    noise_threshold = 3.0 * np.std(residual_centered)
    suspicious = residual_smooth > noise_threshold

    # 4. Perturbation stability test
    # Stable predictions shouldn't change much with small input noise
    perturbation_diffs = []
    for _ in range(num_perturbations):
        noise = np.random.randn(*dl_output.shape) * perturbation_sigma * np.std(dl_output)
        perturbed_output = dl_output + noise  # simulate perturbed DL output
        diff = np.std(perturbed_output - dl_output) / (np.std(dl_output) + 1e-10)
        perturbation_diffs.append(diff)
    stability_score = 1.0 - min(np.mean(perturbation_diffs) / perturbation_sigma, 1.0)

    has_risk = (
        structure_score > structure_threshold
        or np.sum(suspicious) > 0.01 * suspicious.size
    )

    diagnostics = (
        f"Residual structure score: {structure_score:.3f} "
        f"(threshold: {structure_threshold})\n"
        f"Suspicious pixels: {np.sum(suspicious)} "
        f"({100 * np.sum(suspicious) / suspicious.size:.2f}%)\n"
        f"Stability score: {stability_score:.3f}\n"
        f"Correlation (DL vs conventional): "
        f"{pearsonr(dl_norm.ravel(), conv_norm.ravel())[0]:.4f}\n"
    )

    return {
        "residual_structure_score": structure_score,
        "suspicious_regions": suspicious,
        "stability_score": stability_score,
        "has_hallucination_risk": has_risk,
        "diagnostics": diagnostics,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Acquire sufficient data quality to avoid extreme reliance on DL denoising or reconstruction.
- When training DL models, use diverse training data that covers the expected sample variability.
- Validate DL models on held-out test data that includes edge cases and out-of-distribution samples.
- Document the training data distribution and known failure modes of any DL model used in the pipeline.

### Correction — Traditional Methods

The primary safeguard is always to compare DL-processed outputs with conventional reconstructions and inspect residuals for structure.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def hallucination_audit(dl_image, conventional_image, output_path=None):
    """
    Generate a visual hallucination audit comparing DL output with
    conventional reconstruction. Highlights regions where DL output
    deviates significantly from the conventional baseline.

    Parameters
    ----------
    dl_image : np.ndarray
        2D DL-processed image.
    conventional_image : np.ndarray
        2D conventionally reconstructed image.
    output_path : str or None
        Path to save the audit figure. If None, displays interactively.
    """
    # Normalize to same dynamic range
    dl_norm = (dl_image - dl_image.mean()) / (dl_image.std() + 1e-10)
    conv_norm = (conventional_image - conventional_image.mean()) / (
        conventional_image.std() + 1e-10
    )

    # Compute residual
    residual = dl_norm - conv_norm

    # Identify structured deviations above noise floor
    noise_floor = np.std(gaussian_filter(conv_norm, sigma=1) - conv_norm)
    suspicious_mask = np.abs(gaussian_filter(residual, sigma=3)) > 3 * noise_floor

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(conv_norm, cmap="gray")
    axes[0, 0].set_title("Conventional Reconstruction")

    axes[0, 1].imshow(dl_norm, cmap="gray")
    axes[0, 1].set_title("DL Output")

    im_res = axes[1, 0].imshow(residual, cmap="RdBu_r", vmin=-3*noise_floor,
                                vmax=3*noise_floor)
    axes[1, 0].set_title("Residual (DL - Conventional)")
    plt.colorbar(im_res, ax=axes[1, 0])

    axes[1, 1].imshow(dl_norm, cmap="gray")
    axes[1, 1].contour(suspicious_mask, colors="red", linewidths=0.5)
    axes[1, 1].set_title("Suspicious Regions (red contours)")

    pct_suspicious = 100 * np.sum(suspicious_mask) / suspicious_mask.size
    fig.suptitle(
        f"DL Hallucination Audit — {pct_suspicious:.1f}% pixels flagged",
        fontsize=14, fontweight="bold",
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Audit figure saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
```

### Correction — AI/ML Methods

Uncertainty quantification methods provide per-pixel confidence estimates that can flag potential hallucinations:

- **Ensemble methods:** Train multiple networks with different initializations or architectures and compute pixel-wise variance across predictions. High variance indicates unreliable regions.
- **Monte Carlo dropout:** Run inference multiple times with dropout enabled; variance across runs estimates epistemic uncertainty.
- **Learned uncertainty:** Train the network to output both a prediction and an uncertainty map (heteroscedastic loss).
- **Residual analysis:** Compare the DL output against the raw data or a conventional reconstruction; structured residuals indicate hallucination.

```python
import numpy as np


def ensemble_uncertainty(models, input_data, dropout_passes=20):
    """
    Estimate uncertainty via Monte Carlo dropout or model ensemble.

    Parameters
    ----------
    models : list or single model
        If a list, use as an ensemble. If a single model with dropout,
        run multiple stochastic forward passes.
    input_data : np.ndarray
        Input image/sinogram, shape matching model input.
    dropout_passes : int
        Number of stochastic passes (only used for single model with dropout).

    Returns
    -------
    dict with keys:
        'mean_prediction' : np.ndarray — ensemble mean
        'uncertainty_map' : np.ndarray — per-pixel standard deviation
        'high_uncertainty_mask' : np.ndarray — binary mask of unreliable regions
    """
    predictions = []

    if isinstance(models, list):
        # Ensemble approach
        for model in models:
            pred = model.predict(input_data[np.newaxis, ..., np.newaxis])
            predictions.append(pred.squeeze())
    else:
        # MC Dropout approach: model must have dropout layers active
        model = models
        for _ in range(dropout_passes):
            pred = model(input_data[np.newaxis, ..., np.newaxis], training=True)
            predictions.append(pred.numpy().squeeze())

    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)

    # Flag high-uncertainty regions (> 2 std above median uncertainty)
    unc_threshold = np.median(uncertainty) + 2 * np.std(uncertainty)
    high_unc_mask = uncertainty > unc_threshold

    return {
        "mean_prediction": mean_pred,
        "uncertainty_map": uncertainty,
        "high_uncertainty_mask": high_unc_mask,
    }
```

## Impact If Uncorrected

Hallucinated features can lead to entirely incorrect scientific conclusions. In materials science, a hallucinated particle or void could alter measured porosity, grain size distributions, or defect densities. In biological imaging, false structures could be misidentified as organelles or pathological features. In ptychography, hallucinated phase features corrupt quantitative electron density measurements. Because DL hallucinations appear natural and self-consistent, they are far more dangerous than traditional artifacts (rings, streaks) that are at least visually obvious. Published results based on unchecked DL outputs risk irreproducibility and retraction.

## Related Resources

- [TomoGAN denoising](../../03_ai_ml_methods/denoising/tomogan.md) — GAN-based denoising where hallucination risk is particularly high
- [Low-Dose Noise](../tomography/low_dose_noise.md) — the primary use case driving DL denoising (and hallucination risk)
- [Sparse Angle Artifact](../tomography/sparse_angle_artifact.md) — DL-based sparse-view reconstruction is especially hallucination-prone
- [Gottschling et al. 2020 — The troublesome kernel](https://doi.org/10.1137/20M1387237)
- [Antun et al. 2020 — Instabilities of DL in image reconstruction](https://doi.org/10.1073/pnas.1907377117)

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Bhadra et al. 2021](https://doi.org/10.1109/TMI.2021.3077857) | Paper (PMC8673588) | Figs 3--5 | On Hallucinations in Tomographic Image Reconstruction — systematic analysis of DL hallucinations with real CT examples | -- |
| [DIDSR/sfrc (GitHub)](https://github.com/DIDSR/sfrc) | Repository | -- | sFRC tool for detecting hallucinated regions in DL-reconstructed images via spectral Fourier ring correlation | -- |

> **Recommended reference**: [Bhadra et al. 2021 — On Hallucinations in Tomographic Image Reconstruction (IEEE TMI)](https://doi.org/10.1109/TMI.2021.3077857)

## Key Takeaway

Deep learning hallucination is the most scientifically dangerous artifact in modern synchrotron data analysis. Never rely solely on DL-processed results for quantitative conclusions — always perform residual analysis against conventional reconstructions, use uncertainty quantification, and treat DL outputs as hypotheses to be validated rather than ground truth.
