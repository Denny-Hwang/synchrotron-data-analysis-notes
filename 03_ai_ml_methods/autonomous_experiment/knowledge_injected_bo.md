# Knowledge-Injected Bayesian Optimization for Spectroscopy

**Reference**: Du et al., npj Computational Materials (2025)

## Overview

Knowledge-Injected Bayesian Optimization (KI-BO) extends standard BO by incorporating
domain-specific knowledge about spectral structure into the acquisition function. For
XANES spectroscopy, this means the algorithm understands where absorption edges and
pre-edge features are likely to occur, enabling 5x faster data collection while
maintaining spectral accuracy.

## Why Standard BO Falls Short for XANES

Standard BO acquisition functions are agnostic to XANES physics:

- **Variance-only**: Samples uniformly to reduce uncertainty everywhere equally —
  wastes budget on flat pre-edge/post-edge regions
- **UCB (Upper Confidence Bound)**: Designed for maximization, not spectrum
  reconstruction — biases toward high-absorption regions
- **EI (Expected Improvement)**: Seeks to improve on current best — inappropriate
  for spectral mapping where all points matter

## Knowledge-Injected Acquisition Function

```
α(E) = w₁ · |∂μ(E)/∂E| + w₂ · σ(E) + w₃ · P_edge(E)

Components:
  |∂μ(E)/∂E|  = Gradient magnitude from GP posterior mean
                 → High near absorption edge, low in flat regions
  σ(E)        = GP posterior uncertainty
                 → High where data is sparse
  P_edge(E)   = Prior probability of being in the edge region
                 → Encodes domain knowledge about element-specific edge energies

Typical weights: w₁ = 0.5, w₂ = 0.3, w₃ = 0.2
```

### Gradient-Aware Sampling

```
Pre-edge region:    |∂μ/∂E| ≈ 0    → few samples needed
Absorption edge:    |∂μ/∂E| >> 0   → many samples allocated
White line peak:    |∂μ/∂E| varies  → moderate samples
Post-edge (EXAFS):  |∂μ/∂E| ≈ 0   → few samples needed
```

This naturally concentrates measurement budget where it matters most.

## Algorithm

```
1. Seed: Measure 5-10 broadly spaced energy points
2. Fit GP surrogate to observations: f(E) ~ GP(μ, K)
3. Compute knowledge-injected acquisition: α(E) over energy range
4. Select E* = argmax α(E)
5. Move monochromator to E*, collect I₀ and I_t
6. Compute μ(E*) = -ln(I_t / I₀)
7. Update GP with new (E*, μ(E*)) observation
8. If convergence criterion met → stop
   Else → go to step 2
9. Output: GP posterior mean as reconstructed XANES spectrum
```

### Convergence Criterion

```python
# Stop when maximum acquisition value drops below threshold
# or when N_max measurements reached
if max(acquisition_values) < epsilon or n_measurements >= N_max:
    stop_and_interpolate()
```

## Integration with Bluesky

```python
from bluesky import RunEngine
from bluesky.plans import scan
from ophyd import EpicsMotor, EpicsSignalRO

# Monochromator and detectors
mono = EpicsMotor("XF:25IDC-OP:1{Mono}", name="mono_energy")
I0 = EpicsSignalRO("XF:25IDC-BI:1{IC:I0}", name="I0")
It = EpicsSignalRO("XF:25IDC-BI:1{IC:It}", name="It")

# Adaptive plan
def adaptive_xanes(element_edge, E_range, n_seed=8, n_max=50):
    """Knowledge-injected BO XANES plan for Bluesky."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    import numpy as np

    # Seed measurement
    E_seed = np.linspace(E_range[0], E_range[1], n_seed)
    observations = []

    for E in E_seed:
        yield from bps.mv(mono, E)
        ret = yield from bps.trigger_and_read([I0, It])
        mu = -np.log(ret[It.name]["value"] / ret[I0.name]["value"])
        observations.append((E, mu))

    # Adaptive loop
    for step in range(n_max - n_seed):
        X = np.array([o[0] for o in observations]).reshape(-1, 1)
        y = np.array([o[1] for o in observations])

        gp = GaussianProcessRegressor()
        gp.fit(X, y)

        # Knowledge-injected acquisition
        E_candidates = np.linspace(E_range[0], E_range[1], 500).reshape(-1, 1)
        mu_pred, sigma = gp.predict(E_candidates, return_std=True)
        gradient = np.abs(np.gradient(mu_pred.flatten()))
        acq = 0.5 * gradient + 0.3 * sigma + 0.2 * edge_prior(E_candidates, element_edge)

        E_next = E_candidates[np.argmax(acq)][0]
        yield from bps.mv(mono, E_next)
        ret = yield from bps.trigger_and_read([I0, It])
        mu = -np.log(ret[It.name]["value"] / ret[I0.name]["value"])
        observations.append((E_next, mu))

    return observations
```

## Performance

| Metric | Standard Grid | Standard BO | KI-BO |
|--------|--------------|-------------|-------|
| **Points needed** | 200-500 | 60-100 | 30-50 |
| **Edge error (eV)** | < 0.01 | 0.2-0.5 | < 0.1 |
| **White line error** | < 0.01 | 0.1-0.3 | < 0.03 |
| **RMSE** | reference | 0.02 | < 0.005 |
| **Time savings** | — | 2-3x | **5x** |

## Strengths

1. **5x speedup**: Enables time-resolved XANES with 5x finer temporal resolution
2. **Physics-informed**: Domain knowledge prevents sampling errors near critical features
3. **Bluesky-native**: Drops into existing beamline infrastructure
4. **Element-agnostic framework**: Edge prior can be configured for any element
5. **Quantitative accuracy**: Maintains sub-0.1 eV accuracy for edge position

## Limitations

1. **Single-edge focus**: Current implementation targets one absorption edge at a time
2. **GP scaling**: O(N³) for GP fitting, though N is small (30-50 points)
3. **Element-specific prior**: Edge prior needs configuration per element/edge type
4. **No EXAFS support**: k-space sampling strategy needed for EXAFS region
5. **Beam stability assumption**: Requires stable beam during adaptive sequence

## Comparison with Other Autonomous Methods

| Method | Domain | Spatial/Spectral | Decision Type |
|--------|--------|-----------------|---------------|
| **KI-BO XANES** | Spectroscopy | Spectral (energy) | Next energy point |
| **AI-NERD** | XPCS | Temporal | Next measurement time |
| **ROI-Finder** | XRF | Spatial (x, y) | Next scan region |
| **Standard BO** | General | Any | Next parameter set |
