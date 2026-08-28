---
doc_id: REL-E0100
title: "Release Notes — explorer-v0.10.0"
status: draft
version: 0.10.0
last_updated: 2026-08-28
supersedes: null
related: [ADR-008, REL-E090, REL-N110]
---

# Release Notes — explorer-v0.10.0

**Phase R17 — modern-research Lab recipes.** Minor release answering the
review question "why are the Lab's examples all classical?": the Lab's
contract (pure CPU functions, instant slider response, no bundled
weights per ADR-008) rules out pretrained deep-learning models — but
not the *principles* of the modern literature. Three new recipes bring
the 2018–2025 research generation into the Lab, and a metric-fairness
fix makes their results (and everyone else's) score honestly.

## New recipes (14 → 17)

- **Self-Supervised Tuning — Noise2Self J-invariance (Batson 2019)**
  (`experiments/cross_cutting/self_supervised_tuning/`). The flagship
  modern addition: sweeps a classical denoiser's strength and scores
  each setting with the J-invariant self-supervised loss
  (scikit-image's `calibrate_denoiser` — the reference implementation
  of Noise2Self), then denoises at the winner. **No training, no
  weights, no clean reference** — yet on the low-dose samples the
  self-chosen strength lands at the oracle optimum (severe:
  −5.8 → 16.7 dB; medium: 3.7 → 23.7 dB; mild: 10.3 → 26.0 dB vs the
  held-out reference it never saw), and on the clean identity-check
  sample it picks the weakest setting. References wire it to the
  family the reviews cover: Noise2Self 2019, Noise2Inverse 2020, and
  Obata et al.'s 2025 J. Synchrotron Rad. bone-μCT application.
- **Ring Artifact — Fitting-Based Removal (Vo 2018)**
  (`experiments/tomography/ring_artifact_fitting/`): the full-stripe
  member of Vo, Atwood & Drakopoulos 2018 (adapted from Sarepy's
  `remove_stripe_based_fitting`) — polynomial fit per detector column,
  smoothed-surface ratio correction.
- **Ring Artifact — Dead-Stripe Interpolation (Vo 2018)**
  (`experiments/tomography/ring_artifact_interpolation/`): the
  detect-then-replace member for unrecoverable stripes — MAD-robust
  column detection + row-wise linear interpolation. With the existing
  sorting-based and wavelet-FFT recipes, the Lab now covers the full
  Vo 2018 stripe taxonomy that production pipelines (Algotom
  `remove_all_stripe`) combine.

## Fixed

- **Metric normalisation now uses the reference's scale** (R16.1,
  `lib/experiments._normalize_pair`). Previously reference and
  candidate were min-max normalised *independently*, so any algorithm
  that changed the value distribution was scored on a re-stretched
  axis — a strong denoiser that shrinks the range showed a spurious
  "PSNR regressed" banner even when its true MSE against the clean
  reference improved (verified: the self-supervised tuner's loss curve
  tracked true MSE exactly while the old displayed PSNR inverted the
  ranking; the beam-hardening recipe's false "regressed" verdict came
  from the same distortion). Both arrays are now scaled by the
  reference's min/max — a true same-axis comparison. Displayed
  PSNR/SSIM values shift across all recipes; the direction of change
  is now trustworthy.
- Landing-page and static-site recipe counts are computed from
  `experiments/` instead of hand-maintained ("14 recipes" had already
  drifted).

## Changed

- Lab intro now states *why* pretrained-DL recipes are absent (pure-CPU
  contract + ADR-008 no-bundled-weights policy) and points to the
  Self-Supervised Tuning recipe and the 2025–2026 paper reviews for
  the modern-methods story.

## Test plan

- Full suite green (341 tests) including the R16 drift tests, which
  automatically cover the new recipes (recipe↔manifest coverage,
  role semantics, contract execution).
- All three new recipes executed on every declared sample with default
  parameters: all improve or hold PSNR/SSIM on their target samples,
  leave false-positive traps essentially untouched, and run in
  ≤ 2.5 s per invocation on CPU.

Traceability: ADR-008 (Lab contract; no-weights policy), REL-N110
(the 2025–2026 reviews these recipes operationalise), invariant #9
(static stat lines updated via the computed-count mechanism).
