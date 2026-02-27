# AI/ML Methods for X-ray Absorption Spectroscopy

## Overview

AI/ML methods for XAS automate spectral classification, enable rapid chemical speciation,
and improve analysis of complex multi-component mixtures. These approaches are particularly
valuable for large µ-XANES imaging datasets where manual analysis of thousands of
spectra is impractical.

## ML Problem Classification

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| Species identification | Classification | XANES spectrum | Chemical species label |
| Linear combination fitting | Regression | Spectrum + references | Component fractions |
| Edge energy determination | Regression | XANES spectrum | Edge energy (eV) |
| Oxidation state mapping | Classification | µ-XANES image stack | Oxidation state map |
| Spectral denoising | Regression | Noisy spectrum | Clean spectrum |
| Structure prediction | Regression | EXAFS spectrum | Bond distances, coordination |

## Spectral Classification

### Supervised Classification

Train classifiers on libraries of reference XANES spectra:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# X: (Nsamples, Nenergy_points) - normalized XANES spectra
# y: labels (e.g., 'Fe2O3', 'FeO', 'FePO4', ...)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Classify unknown spectrum
prediction = clf.predict(unknown_spectrum.reshape(1, -1))
probabilities = clf.predict_proba(unknown_spectrum.reshape(1, -1))
```

**Approaches**:
- **Random Forest**: Robust, interpretable feature importance
- **SVM**: Effective for high-dimensional spectral data
- **Neural Networks**: MLP or 1D CNN for complex spectral patterns
- **Transfer learning**: Pre-train on simulated spectra (FEFF/FDMNES), fine-tune on experimental

### Database-Driven Classification

Emerging databases of reference spectra enable ML training:
- **XASdb**: Database of experimental XANES spectra
- **Materials Project**: Computed XANES from crystal structures (FEFF calculations)
- **XANESNET**: Neural network trained on computed Fe K-edge spectra

## PCA-Based Analysis

### Component Identification

```python
from sklearn.decomposition import PCA

# For µ-XANES imaging: reshape to (Npixels, Nenergies)
spectra_2d = xanes_stack.reshape(-1, n_energies)

# Remove pixels with no signal
mask = spectra_2d.mean(axis=1) > threshold
spectra_filtered = spectra_2d[mask]

pca = PCA(n_components=10)
scores = pca.fit_transform(spectra_filtered)
components = pca.components_

# Determine number of independent components
# (number of significant eigenvalues)
explained_var = pca.explained_variance_ratio_
n_components = np.argmax(np.cumsum(explained_var) > 0.99) + 1
print(f"Number of independent spectral components: {n_components}")
```

### Target Transformation

Combine PCA with target testing to identify end-member spectra:
1. PCA determines number of independent components
2. Test reference spectra as potential end-members
3. SPOIL/IND values assess fit quality
4. Most commonly used approach for µ-XANES analysis

## Automated Linear Combination Fitting

### Traditional LCF
```python
from scipy.optimize import nnls

# reference_matrix: (Nenergies, Nreferences)
# unknown: (Nenergies,) - the unknown spectrum

fractions, residual = nnls(reference_matrix, unknown)
fractions = fractions / fractions.sum()  # normalize to 100%
```

### ML-Enhanced LCF

**Problems with traditional LCF**:
- Requires pre-selection of reference compounds
- Combinatorial explosion if many possible references
- Cannot handle unknown phases

**ML solutions**:
- **Neural network fitting**: Train NN to directly predict fractions from spectrum
- **Variational inference**: Bayesian approach provides uncertainty on fractions
- **Clustering + LCF**: First cluster pixels by similarity, then fit representative spectra

## Deep Learning for XANES

### 1D Convolutional Neural Networks

```python
import torch.nn as nn

class XANESClassifier(nn.Module):
    def __init__(self, n_energies, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (batch, 1, n_energies)
        features = self.conv(x).squeeze(-1)
        return self.fc(features)
```

### Applications
- **Oxidation state prediction**: Classify Fe²⁺/Fe³⁺ from edge shape
- **Coordination environment**: Predict tetrahedral/octahedral from pre-edge features
- **Speciation mapping**: Apply trained model to each pixel in µ-XANES stack

## Spectral Denoising

### Wavelet Denoising
- Apply wavelet transform, threshold high-frequency coefficients, inverse transform
- Preserves spectral features better than smoothing

### Autoencoder Denoising
- Train autoencoder on clean reference spectra
- Input noisy spectrum → encode → decode → clean spectrum
- Advantage: learns physically meaningful spectral features

## Emerging Methods

### Transfer Learning from Simulations
1. Generate large training set using FEFF/FDMNES calculations from crystal structures
2. Pre-train neural network on simulated spectra
3. Fine-tune on small experimental dataset
4. Addresses the limited experimental training data problem

### Self-Supervised Learning
- Mask portions of spectra during training (masked spectral modeling)
- Learn spectral representations without labels
- Fine-tune for specific classification/regression tasks

### Uncertainty Quantification
- **Monte Carlo dropout**: Run inference multiple times with dropout for uncertainty estimates
- **Ensemble methods**: Train multiple models, use variance as uncertainty
- **Bayesian neural networks**: Direct uncertainty from posterior distribution
- Critical for scientific applications where confidence matters

## Current Limitations

1. **Reference database completeness**: Many environmental species lack reference spectra
2. **Spectral similarity**: Some species are nearly indistinguishable by XANES alone
3. **Non-linear mixing**: Actual spectra may not be simple linear combinations
4. **Self-absorption**: Thick/concentrated samples distort spectral shape
5. **Radiation damage**: Beam damage changes speciation during measurement

## Improvement Opportunities

1. **Comprehensive spectral databases**: Community-wide effort to build labeled databases
2. **Multi-edge analysis**: Combine information from multiple element edges
3. **Joint XAS + XRF**: Use spatial context from XRF to constrain spectral fitting
4. **Active learning**: ML guides which spectra to measure for maximum information
5. **Real-time speciation**: Stream XAS data to ML model during measurement
