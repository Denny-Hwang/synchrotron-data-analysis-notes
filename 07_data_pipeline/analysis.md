# Analysis

## Overview

The analysis stage applies trained machine learning models to processed data,
provides interactive visualization for human interpretation, and enforces
quality gates before results are committed to the archive.

## ML Inference Pipeline

The inference pipeline is deployed as a containerized service that accepts
processed volumes and returns predictions or derived quantities.

```
Processed Volume (Zarr / HDF5)
    |
    v
Model Registry (MLflow)  -->  Inference Service (Triton)
                                    |
                                    v
                              Predictions (JSON / HDF5)
```

### Model Registry

All production models are tracked in MLflow:

| Field | Example |
|---|---|
| Model name | `tomoseg-unet3d-v4` |
| Framework | PyTorch 2.2 |
| Training dataset | `2024Q3-battery-cathode` |
| Validation metric | Dice = 0.94, IoU = 0.89 |
| Input spec | float32, 128^3 patches |

Models are promoted through stages: `Staging` -> `Production` -> `Archived`.
Only `Production` models are used in automated pipeline runs.

### Inference Modes

| Mode | Trigger | Latency | Use Case |
|---|---|---|---|
| Real-time | ZMQ frame arrival | < 500 ms | Live classification during scan |
| Batch | Globus transfer complete | Minutes | Full-volume segmentation |
| On-demand | User request via API | Seconds | Re-analysis with new model |

### Supported Model Types

- **Segmentation**: 3D U-Net, nnU-Net for multi-phase labeling
- **Classification**: ResNet-50 for scan quality / sample type
- **Anomaly detection**: Autoencoder for outlier frame detection
- **Super-resolution**: ESRGAN for enhancing low-dose acquisitions

## Visualization Tools

### Jupyter Notebooks

JupyterHub is deployed on ALCF Polaris nodes with direct Eagle filesystem access.
Standard notebooks are provided for common workflows:

| Notebook | Purpose |
|---|---|
| `01_quick_look.ipynb` | Load reconstruction, plot orthogonal slices |
| `02_histogram_analysis.ipynb` | Intensity histograms, threshold selection |
| `03_3d_rendering.ipynb` | Volume rendering with ipyvolume / itkwidgets |
| `04_comparison.ipynb` | Side-by-side before/after processing |

### Streamlit Dashboards

Interactive web dashboards for guided analysis workflows:

- **Scan Browser** -- Browse recent scans, view thumbnails, launch analysis
- **Quality Dashboard** -- Display QA metrics across all scans in a campaign
- **Parameter Explorer** -- Adjust reconstruction / segmentation parameters
- **Report Generator** -- Produce PDF summary reports for each dataset

### Napari

Napari provides GPU-accelerated multi-dimensional image viewing with Dask-backed
lazy loading for out-of-core volumes, layer-based overlays, and manual
annotation tools for correcting segmentation labels.

```python
import napari
import dask.array as da

volume = da.from_zarr("/eagle/eBERlight/recon/scan_0042.zarr")
labels = da.from_zarr("/eagle/eBERlight/seg/scan_0042_labels.zarr")

viewer = napari.Viewer()
viewer.add_image(volume, name="reconstruction", contrast_limits=[0, 0.005])
viewer.add_labels(labels, name="segmentation", opacity=0.4)
napari.run()
```

## Result Validation Process

### Automated Quality Gates

Every processed dataset passes through automated quality checks:

| Check | Criterion | Action on Failure |
|---|---|---|
| Reconstruction completeness | All slices present, no NaN | Reject, re-queue |
| SNR threshold | SNR > 15 dB in ROI | Flag for review |
| Segmentation coverage | > 95% of volume classified | Flag for review |
| Ring artifact score | FFT ring metric < 0.02 | Re-run with stronger filter |
| Model confidence | Mean softmax > 0.85 | Flag low-confidence regions |

### Human Review Workflow

Datasets flagged by automated checks enter a human review queue:

1. **Reviewer** opens the dataset in Napari or Jupyter via a review link.
2. **Annotations** mark regions as accepted, corrected, or rejected.
3. **Decision** is recorded: `accept`, `reprocess`, or `discard`.
4. Accepted datasets proceed to [storage.md](storage.md) for archival.
5. Corrected labels are fed back into the training pipeline to improve models.

### Provenance Tracking

Every analysis action is logged in a provenance record embedded in NeXus metadata:

```json
{
  "scan_id": "scan_0042",
  "pipeline_version": "2.4.1",
  "model": "tomoseg-unet3d-v4",
  "model_hash": "sha256:a3f8c1...",
  "qa_checks": {"snr_db": 22.4, "seg_coverage": 0.987, "ring_score": 0.008},
  "reviewer": "jsmith",
  "decision": "accept",
  "timestamp": "2025-11-03T14:22:00Z"
}
```

## Integration Points

- **Upstream**: Receives processed volumes from [processing.md](processing.md)
- **Downstream**: Validated results proceed to [storage.md](storage.md)
- **Feedback loop**: Corrected labels return to `06_ml_ai/` for model retraining
