# AI-NERD: Unsupervised Fingerprinting for XPCS Dynamics

**Reference**: Horwath et al., Nature Communications (2024)

## Overview

**AI-NERD** (AI for Nonequilibrium Relaxation Dynamics) is an unsupervised machine
learning framework that "fingerprints" material dynamics from X-ray Photon Correlation
Spectroscopy (XPCS) data without requiring physical model assumptions.

## Motivation

Traditional XPCS analysis involves fitting correlation functions to assumed models
(exponential, stretched exponential, etc.). This approach:
- Requires choosing the correct model a priori
- May miss unexpected dynamical behaviors
- Cannot easily classify dynamical states across large datasets
- Is labor-intensive for large parameter studies

AI-NERD addresses these limitations with a model-free, unsupervised approach.

## Method

### Pipeline

```mermaid
graph TD
    A[XPCS Time Series<br>Speckle patterns] --> B[Correlation Functions<br>g₂(q,τ) at multiple q]
    B --> C[Feature Engineering<br>Flatten or encode g₂ curves]
    C --> D[Dimensionality Reduction<br>UMAP embedding]
    D --> E[Clustering<br>HDBSCAN]
    E --> F[Dynamical Fingerprint Map<br>Each cluster = distinct state]
```

### Step 1: Compute Correlation Functions

```python
# Standard multi-tau correlation algorithm
# For each q bin, compute g₂(τ):
# g₂(q, τ) = <I(q,t) × I(q,t+τ)> / <I(q,t)>²
```

### Step 2: Feature Representation

**Approach A: Direct concatenation**
```python
# Concatenate g₂ curves at all q values into single feature vector
# feature = [g₂(q₁,τ₁), g₂(q₁,τ₂), ..., g₂(qN,τM)]
# Dimension: N_q × N_tau
features = g2_array.reshape(n_measurements, -1)
```

**Approach B: Autoencoder encoding**
```python
# Train autoencoder to compress g₂ curves
# feature = encoder(g₂_curves)  → compact latent representation
```

### Step 3: UMAP Dimensionality Reduction

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)
embedding = reducer.fit_transform(features)
```

UMAP preserves both local and global structure, revealing clusters of similar dynamics.

### Step 4: HDBSCAN Clustering

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(embedding)

# labels: -1 = noise, 0,1,2,... = cluster IDs
# Each cluster represents a distinct dynamical state
```

### Step 5: Interpretation

```python
# For each cluster, compute representative g₂ curve
for cluster_id in range(max(labels) + 1):
    mask = labels == cluster_id
    mean_g2 = features[mask].mean(axis=0).reshape(n_q, n_tau)

    # Visualize: what does this dynamical state look like?
    # Compare to known dynamical models
    # Identify transitions between states (temporal evolution)
```

## Key Innovation

### Model-Free Analysis
- No assumption about relaxation model (exponential, power-law, etc.)
- Discovers dynamical states directly from data structure
- Can identify unexpected or novel dynamical behaviors
- Particularly valuable for nonequilibrium systems where no model exists

### Comprehensive q-Dependence
- Uses full q-dependent g₂ information (not just single-q fitting)
- Captures q-dependent dynamics that single-q analysis would miss
- Fingerprint encodes complete dynamical information

### Temporal Evolution Tracking
```
Time → Cluster sequence: [A, A, A, B, B, B, C, C, B, B, A, A]

Transitions between clusters indicate dynamical phase changes
```

## Application Examples

### Colloidal Gel Aging
- **Observation**: UMAP reveals 3 distinct clusters during gel aging
- **Interpretation**: Liquid → gel transition → aging gel
- **Insight**: Transition time precisely identified without model fitting

### Temperature-Dependent Dynamics
- Map XPCS measurements at different temperatures
- Clusters correspond to dynamical phases (crystalline, liquid, glass)
- Phase diagram emerges from unsupervised analysis

### Radiation Damage Monitoring
- Detect onset of beam damage as change in dynamical fingerprint
- Automated detection: flag when fingerprint changes significantly
- Enables adaptive dose management

## Strengths

1. **Model-free**: No physical model assumptions required
2. **Unsupervised**: No labeled training data needed
3. **Comprehensive**: Uses full q-dependent dynamical information
4. **Sensitive**: Detects subtle transitions missed by standard analysis
5. **Scalable**: Handles large parameter studies automatically
6. **Interpretable**: Cluster centers provide representative dynamics

## Limitations

1. **Post-hoc**: Currently applied after data collection (not real-time)
2. **Hyperparameters**: UMAP and HDBSCAN have tunable parameters
3. **Feature engineering**: Choice of representation affects results
4. **Physical interpretation**: Clusters need expert interpretation
5. **Validation**: Difficult to validate without ground truth labels

## Relevance to eBERlight

### APS-U XPCS (12-ID-E)
- Post-upgrade XPCS generates massive datasets (millions of frames)
- Traditional analysis cannot keep up with data rates
- AI-NERD enables automated classification of dynamical states
- Real-time implementation would enable autonomous experiment steering

### Extension to Other Modalities
- Principle applicable beyond XPCS:
  - Time-resolved SAXS → fingerprint structural evolution
  - Sequential XANES spectra → fingerprint chemical changes
  - Time-series XRF maps → fingerprint elemental redistribution
