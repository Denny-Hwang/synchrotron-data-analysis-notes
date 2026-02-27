# Autonomous Experiment Steering

## Overview

Autonomous experiment methods use AI/ML to make real-time decisions during synchrotron
experiments — selecting measurement regions, optimizing parameters, and adapting
acquisition strategies based on incoming data.

## Why Autonomous?

Post APS-U challenges driving autonomous experimentation:
1. **Data volume**: Human cannot review TB-scale data in real-time
2. **Beam time scarcity**: Maximize scientific output per hour of beam time
3. **Complexity**: Multi-parameter optimization exceeds human capacity
4. **Reproducibility**: Algorithmic decisions are reproducible and documented

## Autonomy Levels

| Level | Description | Human Role | Example |
|-------|-------------|-----------|---------|
| **0** | Manual | All decisions | Traditional experiment |
| **1** | Advisory | Final decision | ROI-Finder suggests cells |
| **2** | Supervised autonomous | Override capability | Auto-alignment with approval |
| **3** | Conditional autonomous | Set boundaries | Autonomous scanning within area |
| **4** | Full autonomous | Monitor only | Self-driving beamline |

Current state: Most systems at Level 1-2; research pushing toward Level 3-4.

## Method Categories

| Method | Decision Type | Input | Action |
|--------|--------------|-------|--------|
| **ROI selection** | Where to measure | Coarse data | Prioritize regions |
| **Parameter optimization** | How to measure | Quality metrics | Adjust exposure, energy |
| **Adaptive scanning** | What resolution | Running results | Vary step size |
| **Anomaly detection** | When to flag | Data stream | Alert operator |
| **Stopping criteria** | When to stop | Cumulative data | End acquisition |

## Directory Contents

| File | Content |
|------|---------|
| [roi_finder.md](roi_finder.md) | ML-guided ROI selection for XRF |
| [bayesian_optimization.md](bayesian_optimization.md) | Bayesian parameter optimization |
| [ai_nerd.md](ai_nerd.md) | Unsupervised dynamics fingerprinting |
