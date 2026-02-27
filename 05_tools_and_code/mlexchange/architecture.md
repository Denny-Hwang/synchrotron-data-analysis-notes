# MLExchange -- Microservice Architecture

## High-Level Architecture

```
Browser (React GUI)
  |
  v
API Gateway (FastAPI)
  |
  +---> Auth Service (user management, tokens)
  +---> Data Service (upload, storage, metadata)
  +---> Model Service (model zoo registry)
  +---> Training Service (job submission, monitoring)
  +---> Inference Service (batch / streaming prediction)
  +---> Compute Service (resource allocation)
  |
  v
Compute Back-ends
  +---> Local GPU (Docker containers)
  +---> NERSC / HPC (Slurm via job proxy)
  +---> Kubernetes cluster
```

## Microservice Descriptions

### API Gateway

- Single entry point for all client requests.
- Routes to downstream services based on URL prefix.
- Handles CORS, rate limiting, and request logging.

### Data Service

- Manages upload, storage, and retrieval of datasets.
- Stores data in a shared filesystem or object store (MinIO / S3).
- Tracks dataset metadata (shape, dtype, provenance) in a MongoDB collection.

### Model Service

- Maintains a registry of available ML models (the "model zoo").
- Each model entry includes: architecture class, default hyperparameters,
  expected input/output shapes, and a Docker image reference.
- Models are versioned and can be published or kept private.

### Training Service

- Accepts training job requests (dataset ID, model ID, hyperparameters).
- Submits jobs to the Compute Service and polls for status.
- Stores training logs, metrics (loss curves), and model checkpoints.

### Inference Service

- Loads a trained model checkpoint and runs prediction on new data.
- Supports batch mode (full dataset) and streaming mode (per-image via
  message queue).

### Compute Service

- Abstracts compute back-ends behind a uniform job submission API.
- For local GPU: launches Docker containers with GPU passthrough.
- For HPC: generates Slurm job scripts and submits via SSH or REST proxy.
- For Kubernetes: creates Job or Deployment resources via the K8s API.

## DLSIA Integration

DLSIA (Deep Learning for Scientific Image Analysis) is the default model
library used by MLExchange. It provides:

- **TUNet** -- tuneable U-Net with configurable depth, width, and skip
  connections.
- **TUNet3+** -- dense skip variant for multi-scale feature fusion.
- **SimCLR encoder** -- self-supervised pre-training for downstream tasks.

Models are defined in PyTorch and wrapped in a standard `DLSIAModel` class
that MLExchange Training and Inference services call via a uniform
`fit()` / `predict()` interface.

## Message Queue

- RabbitMQ (or Redis Streams) connects services for asynchronous event
  processing.
- Training completion events trigger automatic inference on validation sets.
- Results are pushed back to the Data Service for visualization in the GUI.

## Deployment

- Each microservice is packaged as a Docker image.
- Docker Compose for local development; Helm charts for Kubernetes
  production deployment.
- Configuration via environment variables and a shared `config.yaml`.
