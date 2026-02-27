# Bayesian Optimization for Synchrotron Experiments

## Overview

Bayesian Optimization (BO) is a sample-efficient optimization strategy for
experimentparameters where each evaluation is expensive (e.g., requires beam time).
It builds a probabilistic surrogate model of the objective function and uses an
acquisition function to decide where to sample next.

## Why Bayesian Optimization?

Synchrotron experiments face optimization challenges:
- **Expensive evaluations**: Each measurement takes minutes to hours
- **Limited budget**: Beam time is scarce (typically 3-5 days)
- **High-dimensional**: Multiple parameters interact (energy, exposure, position, temperature)
- **Noisy**: Measurements have inherent statistical noise

BO is ideal because it minimizes the number of evaluations needed to find optimal parameters.

## Method

### Core Algorithm

```
Initialize: Sample a few random parameter configurations
Repeat until budget exhausted:
    1. Fit surrogate model (GP) to all observations
    2. Compute acquisition function over parameter space
    3. Select next parameters x* = argmax(acquisition)
    4. Evaluate objective f(x*) — run experiment
    5. Add (x*, f(x*)) to observation set
```

### Gaussian Process Surrogate

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# GP models the objective as a random function
# Mean: expected value of objective at any point
# Variance: uncertainty (high where few observations)

kernel = Matern(nu=2.5, length_scale_bounds=(0.01, 10.0))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_observed, y_observed)

# Predict at new point with uncertainty
mu, sigma = gp.predict(X_new, return_std=True)
```

### Acquisition Functions

| Function | Formula | Strategy |
|----------|---------|----------|
| **Expected Improvement (EI)** | E[max(f(x) - f_best, 0)] | Balance exploration/exploitation |
| **Upper Confidence Bound (UCB)** | µ(x) + κ·σ(x) | Tunable exploration parameter |
| **Probability of Improvement (PI)** | P(f(x) > f_best + ξ) | Conservative |
| **Thompson Sampling** | Sample from GP posterior | Randomized exploration |

## Applications in Synchrotron Science

### 1. Exposure Optimization

```
Objective: Maximize SNR / measurement time
Parameters: [exposure_time, detector_distance, energy]
Constraint: Dose < max_allowed_dose

BO finds optimal trade-off between signal quality and speed
```

### 2. Scan Strategy Optimization

```
Objective: Maximize reconstruction quality (SSIM or PSNR)
Parameters: [n_projections, angular_range, resolution]
Constraint: Total scan time < T_max

BO determines minimum projections needed for acceptable quality
```

### 3. Sample Environment Parameters

```
Objective: Maximize feature contrast or scientific signal
Parameters: [temperature, humidity, flow_rate, pH]
Constraint: Sample integrity

BO navigates parameter space to find conditions revealing target phenomena
```

### 4. Energy Selection for XANES

```
Objective: Maximize species discrimination
Parameters: [energy_1, energy_2, energy_3]
Constraint: Use only 3 energy points (for fast ratio mapping)

BO selects optimal energies for multi-energy chemical speciation mapping
```

## Implementation Example

```python
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import torch

# Define bounds for parameters
bounds = torch.tensor([
    [0.001, 0.1, 8.0],   # lower: [exposure_s, step_um, energy_keV]
    [1.0, 10.0, 20.0]    # upper
])

# Initial random samples
X_init = torch.rand(5, 3) * (bounds[1] - bounds[0]) + bounds[0]
Y_init = torch.tensor([run_experiment(x) for x in X_init]).unsqueeze(-1)

# BO loop
for iteration in range(20):
    # Fit GP
    gp = SingleTaskGP(X_init, Y_init)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Acquisition function
    EI = ExpectedImprovement(gp, best_f=Y_init.max())

    # Optimize acquisition
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds, q=1, num_restarts=10, raw_samples=512
    )

    # Run experiment with selected parameters
    new_y = run_experiment(candidate.squeeze())

    # Update dataset
    X_init = torch.cat([X_init, candidate])
    Y_init = torch.cat([Y_init, new_y.unsqueeze(0).unsqueeze(-1)])

    print(f"Iteration {iteration}: best = {Y_init.max():.4f}")
```

## Multi-Objective Bayesian Optimization

Often multiple competing objectives exist:

```
Objective 1: Maximize image quality (resolution, SNR)
Objective 2: Minimize dose (protect sample)
Objective 3: Minimize time (maximize throughput)

→ Pareto-optimal trade-off surface
```

```python
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement

# EHVI acquisition for multi-objective optimization
# Finds the Pareto front efficiently
```

## Strengths

1. **Sample efficient**: Finds optima in 10-50 evaluations (vs. 100s for grid search)
2. **Handles noise**: GP naturally models measurement uncertainty
3. **Any objective**: Black-box optimization — no gradient needed
4. **Uncertainty-aware**: Exploration-exploitation balance
5. **Multi-objective**: Can optimize competing goals simultaneously

## Limitations

1. **Scalability**: Standard GP scales O(N³) — limited to ~1000 observations
2. **Dimensionality**: Works best in low dimensions (< 20 parameters)
3. **Discrete parameters**: Requires extensions for discrete choices
4. **GP assumptions**: Stationarity, smoothness may not always hold
5. **Implementation complexity**: Requires ML expertise to configure properly
