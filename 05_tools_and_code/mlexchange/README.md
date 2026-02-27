# MLExchange

## Overview

MLExchange is a web-based machine learning platform developed at Lawrence
Berkeley National Laboratory (LBNL) for the Advanced Light Source (ALS). It
provides beamline scientists with a low-barrier interface for training,
deploying, and running ML models on experimental data without writing code.

The platform is designed to be facility-agnostic and is being evaluated for
adoption at the Advanced Photon Source (APS) as part of the BER program
initiative.

## Key Features

- **Web GUI** -- browser-based interface for data upload, model selection,
  training configuration, and result visualization.
- **Model zoo** -- pre-built models for common tasks: image segmentation,
  denoising, anomaly detection, and classification.
- **DLSIA integration** -- uses the Deep Learning for Scientific Image
  Analysis (DLSIA) library for flexible U-Net and encoder-decoder
  architectures.
- **Compute abstraction** -- jobs can run on local GPUs, NERSC, or
  Kubernetes clusters without user-side configuration changes.
- **REST API** -- all GUI actions are backed by a REST API, enabling
  programmatic access and pipeline integration.

## Relevance to APS / BER Program

- XRF map segmentation and denoising are natural MLExchange use cases.
- The microservice architecture can be deployed on APS computing
  infrastructure alongside Bluesky/EPICS.
- Model training on ALS data can transfer to APS data with minimal
  retraining, given similar detector geometries.

## Repository

- GitHub: <https://github.com/mlexchange>
- Documentation: <https://mlexchange.readthedocs.io>
- Primary team: CAMERA group, LBNL

## Related Documents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Microservice architecture details |
| [pros_cons.md](pros_cons.md) | Scalability and usability assessment |
