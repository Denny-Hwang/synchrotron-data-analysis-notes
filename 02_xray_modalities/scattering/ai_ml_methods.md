# AI/ML Methods for X-ray Scattering

## Overview

AI/ML methods for scattering focus on automated phase identification, pattern
classification, real-time data analysis, and extraction of dynamics from correlation
data. The high throughput of scattering experiments and the complexity of multi-component
fitting make ML approaches particularly valuable.

## ML Problem Classification

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| Phase identification | Classification | 1D I(q) or 2D pattern | Phase/mixture label |
| Material fingerprinting | Unsupervised | I(q) profiles or g₂(τ) | Cluster assignments |
| Parameter extraction | Regression | I(q) profile | Size, shape parameters |
| Model fitting | Regression | I(q) profile | Best-fit model + params |
| Anomaly detection | Classification | Time-resolved I(q) | Transition events |
| Dynamics classification | Classification | g₂(q,τ) | Dynamics type |

## AI-NERD: Unsupervised Fingerprinting for XPCS

**Reference**: Horwath et al., Nature Communications (2024)

### Concept
**AI-NERD** (AI for Nonequilibrium Relaxation Dynamics) is an unsupervised ML method
that "fingerprints" material dynamics from XPCS correlation functions without requiring
physical model fitting.

### Method
```
XPCS correlation functions g₂(q, τ)
    │
    ├─→ Feature extraction
    │       Encode g₂ at multiple q values into fixed-length feature vector
    │       Options: direct sampling, wavelet coefficients, autoencoder
    │
    ├─→ Dimensionality reduction
    │       UMAP or t-SNE → 2D embedding
    │
    ├─→ Clustering
    │       HDBSCAN or k-means on embedded space
    │
    └─→ Fingerprint map
            Each cluster = distinct dynamical state
            Changes in cluster = dynamical transitions
```

### Key Innovation
- **Model-free**: No assumption about functional form of relaxation
- **Unsupervised**: Discovers dynamical states without labels
- **Comprehensive**: Captures full q-dependence of dynamics
- **Sensitive**: Detects subtle transitions that manual fitting might miss

### Applications
- Colloidal gel aging → identify distinct dynamical phases
- Temperature-driven phase transitions in soft matter
- Radiation damage tracking during measurement
- Real-time experiment steering based on detected transitions

*See detailed review: [04_publications/ai_ml_synchrotron/review_ai_nerd_2024.md](../../04_publications/ai_ml_synchrotron/review_ai_nerd_2024.md)*

## Phase Identification from Scattering

### Traditional Approach
- Compare measured I(q) to database of known patterns
- Rietveld refinement for crystalline phases (WAXS)
- Manual selection and fitting of SAXS models

### ML Approaches

#### 1. CNN on 2D Patterns
```python
# Input: 2D scattering pattern (256×256 pixels)
# Output: Phase classification (N classes)

class ScatteringClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4)
        )
        self.classifier = nn.Linear(128 * 16, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
```

#### 2. 1D Profile Matching
- Train on library of simulated I(q) profiles for known structures
- Random Forest, SVM, or MLP classifiers on binned I(q) values
- Handle mixtures with multi-label classification

#### 3. Variational Autoencoders (VAE)
- Learn latent representation of scattering patterns
- Latent space structure reveals material categories
- Generate synthetic patterns for data augmentation

## SAXS Model Fitting

### Traditional
Fit analytical models (sphere, cylinder, core-shell, etc.) to I(q):

```python
# SASView-style model fitting
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment

model = load_model('sphere')
# Parameters: radius, sld, sld_solvent, scale, background
# Minimize chi² between model and data
```

### Neural Network Fitting

Train NN to directly predict model parameters from I(q):

```
I(q) profile → CNN/MLP → [radius, polydispersity, concentration, ...]
```

**Advantages**:
- Instantaneous inference (~ms vs. minutes for least-squares fitting)
- Can handle degenerate solutions by predicting probability distributions
- Enables real-time analysis during high-throughput measurements

**Training data**: Generate large library from known models with varied parameters

## Time-Resolved Analysis

### Kinetic SAXS
Monitor structural changes during reactions, self-assembly, or phase transitions:

```
Time series of I(q,t) profiles
    │
    ├─→ SVD / PCA decomposition → identify number of independent components
    │
    ├─→ MCR-ALS (Multivariate Curve Resolution) → extract pure component spectra
    │
    └─→ ML classification → detect transition points, classify phases
```

### Automated Event Detection
- CNN trained to identify transition events in time-resolved SAXS
- Flags frames where significant structural changes occur
- Enables autonomous experiment decisions (e.g., adjust temperature ramp rate)

## XPCS Dynamics Analysis

### Traditional Fitting
```python
from scipy.optimize import curve_fit

def kww(tau, beta, tau_r, gamma):
    """Kohlrausch-Williams-Watts stretched exponential."""
    return 1 + beta * np.exp(-2 * (tau / tau_r)**gamma)

# Fit each q independently
for q_idx in range(n_q):
    popt, pcov = curve_fit(kww, tau, g2[q_idx],
                           p0=[0.1, 1.0, 1.0],
                           bounds=([0, 0, 0], [1, np.inf, 2]))
```

### ML-Enhanced XPCS Analysis

1. **Relaxation time extraction**: CNN predicts τ(q) directly from g₂ curves
2. **Dynamics classification**: Categorize dynamics type (diffusive, ballistic, compressed, stretched)
3. **q-dependent analysis**: Predict dispersion relation D(q) from full g₂(q,τ) dataset
4. **Two-time analysis**: CNN identifies aging regimes from two-time correlation maps

## Benchmark Datasets

| Dataset | Type | Description | Access |
|---------|------|-------------|--------|
| SAXS-benchmark | SAXS | Synthetic + experimental profiles | Community |
| NIST SRM standards | SAXS/WAXS | Reference materials (glassy carbon, silver behenate) | NIST |
| XPCS colloidal gels | XPCS | Time-resolved dynamics datasets | APS |

## Current Limitations

1. **Training data**: Limited labeled scattering datasets; heavy reliance on simulations
2. **Model generalization**: Networks trained on specific material systems may not transfer
3. **Multi-component mixtures**: Deconvolution of scattering from complex mixtures remains challenging
4. **Absolute scaling**: ML methods often ignore absolute intensity information
5. **q-resolution effects**: Instrumental smearing must be accounted for

## Improvement Opportunities

1. **Self-supervised pre-training**: Learn scattering representations from unlabeled data
2. **Physics-informed networks**: Incorporate scattering theory constraints into architecture
3. **Generative models**: VAE/diffusion models for exploring structure-scattering relationships
4. **Multi-modal**: Combined SAXS + WAXS + XRF analysis
5. **Streaming analysis**: Real-time phase identification during high-throughput measurements
6. **Foundation models**: Large pre-trained models for scattering data across material classes
