# MLExchange -- Scalability and Usability Assessment

## Strengths

### Usability

- **Low barrier to entry** -- the web GUI allows beamline scientists to run
  ML models without writing Python code or managing environments.
- **Pre-built model zoo** -- common tasks (segmentation, denoising) are
  available out of the box, reducing time to first result.
- **Visual feedback** -- training loss curves, prediction overlays, and
  confusion matrices are rendered in the browser.
- **REST API** -- advanced users can script workflows programmatically,
  integrating MLExchange into automated pipelines.

### Scalability

- **Microservice design** -- services can be scaled independently; adding
  GPU workers does not require changes to the API gateway or GUI.
- **Multi-backend compute** -- jobs can be routed to local GPUs, NERSC HPC,
  or Kubernetes clusters based on dataset size and urgency.
- **Containerised models** -- each model runs in its own Docker container,
  isolating dependencies and enabling reproducibility.

### Transferability

- **Facility-agnostic** -- originally built for ALS but designed without
  ALS-specific assumptions, making it deployable at APS.
- **DLSIA models generalise** -- architectures trained on ALS ptychography
  or tomography data have been successfully fine-tuned on APS XRF data.

## Weaknesses

### Usability

- **Limited model customisation in GUI** -- advanced architecture changes
  require editing Python code outside the platform.
- **Documentation gaps** -- API reference and deployment guides are still
  evolving; some endpoints lack examples.
- **No notebook integration** -- Jupyter users must switch contexts to the
  web GUI rather than calling MLExchange from a notebook cell.

### Scalability

- **Single-node default** -- the Docker Compose deployment targets a single
  machine; scaling to a multi-node Kubernetes cluster requires additional
  ops expertise.
- **Large dataset bottleneck** -- datasets must be uploaded to the Data
  Service before training; for multi-GB tomography volumes this introduces
  latency.
- **No streaming training** -- models cannot train on data as it arrives
  from the detector; the full dataset must be available upfront.

### Operational

- **Dependency on external services** -- requires MongoDB, RabbitMQ, and a
  container runtime, increasing deployment complexity.
- **Authentication immaturity** -- the auth service currently supports basic
  token auth; integration with facility identity providers (e.g., Globus
  Auth, LDAP) is in progress.

## Recommendations for APS Adoption

1. Deploy on an APS Kubernetes cluster with persistent GPU nodes.
2. Integrate the Data Service with the APS Globus data fabric to avoid
   redundant uploads.
3. Add a Jupyter extension or `mlexchange-client` Python package so users
   can submit jobs from notebooks.
4. Work with the MLExchange team to add Globus Auth / APS LDAP support.
5. Contribute APS-specific model templates (XRF segmentation, tomographic
   denoising) to the model zoo.
