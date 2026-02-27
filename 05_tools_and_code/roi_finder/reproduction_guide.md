# ROI-Finder: Reproduction Guide

## Prerequisites

- Python 3.8+ (3.8 recommended for compatibility)
- Conda or pip package manager
- ~2 GB disk space (code + sample data)
- No GPU required

## Step 1: Environment Setup

```bash
# Option A: Conda (recommended)
conda create -n roifinder python=3.8
conda activate roifinder

# Option B: venv
python -m venv roifinder-env
source roifinder-env/bin/activate  # Linux/Mac
```

## Step 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/arshadzahangirchowdhury/ROI-Finder.git
cd ROI-Finder

# Install dependencies
pip install -r requirements.txt

# Or install as package (if setup.py available)
pip install -e .
```

## Step 3: Obtain Sample Data

Check the repository for sample data:

```bash
# Sample data may be in the repository
ls data/

# Or download from linked sources (check README)
# Typical sample: MAPS-processed HDF5 file with multi-element XRF maps
```

If no sample data is included, you can use any MAPS-format HDF5 file from your
beamline experiments or request sample data from the paper authors.

## Step 4: Run the Pipeline

### Automated Pipeline

```bash
# Check for example scripts
python examples/run_pipeline.py --input data/sample_xrf.h5
```

### Manual Step-by-Step

```python
# In Python or Jupyter notebook:

# 1. Load data
from roifinder.data_loader import load_xrf_data

data = load_xrf_data('data/sample_xrf.h5')
print(f"Elements: {list(data.keys())}")
print(f"Map shape: {list(data.values())[0].shape}")

# 2. Segment cells
from roifinder.segmenter import segment_cells

labels, props = segment_cells(data['Zn'], min_area=50, max_area=5000)
print(f"Found {labels.max()} cells")

# 3. Extract features and cluster
from roifinder.recommender import extract_features, cluster_cells

features, cell_ids = extract_features(data, labels, elements=['P','S','K','Ca','Fe','Zn'])
cluster_labels, membership, pca_scores, centers = cluster_cells(features, n_clusters=5)

# 4. Get recommendations
from roifinder.recommender import recommend_rois

recommended = recommend_rois(membership, cluster_labels, n_per_cluster=3)
print(f"Recommended {len(recommended)} cells for detailed scanning")

# 5. Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(data['Zn'], cmap='viridis')
axes[0].set_title('Zn Map')
axes[1].imshow(labels, cmap='nipy_spectral')
axes[1].set_title('Segmented Cells')
scatter = axes[2].scatter(pca_scores[:, 0], pca_scores[:, 1], c=cluster_labels, cmap='tab10')
axes[2].set_title('PCA + Clusters')
plt.colorbar(scatter, ax=axes[2])
plt.tight_layout()
plt.savefig('roi_finder_results.png', dpi=150)
```

## Step 5: Launch Interactive GUI

```bash
# If GUI module is available
python -m roifinder.annotator --input data/sample_xrf.h5
```

## Expected Results

### Segmentation
- Clear cell boundaries on high-contrast elemental channels (Zn, Fe, P)
- Cell count typically 50-500 per map (depends on sample and scan area)
- Some over/under-segmentation expected (adjust parameters)

### Clustering
- 3-7 clusters typically meaningful for biological samples
- FPC (Fuzzy Partition Coefficient) > 0.5 indicates reasonable clustering
- PCA typically captures >80% variance in top 3 components

### Recommendation
- Diverse selection across clusters
- Outlier cells (unusual composition) ranked highly
- Results should match domain expert intuition

## Troubleshooting

| Issue | Solution |
|-------|----------|
| HDF5 path error | Check MAPS output format; paths may differ between MAPS versions |
| No cells found | Adjust min_area, max_area, or try different channel |
| Too many clusters | Reduce n_clusters; check FPC for optimal k |
| GUI doesn't launch | Ensure tkinter is installed: `sudo apt install python3-tk` |
| Memory error | Subsample large maps or process in regions |

## Reproducing Paper Results

To reproduce specific results from Chowdhury et al. (2022):

1. Use the exact dataset referenced in the paper (request from authors if needed)
2. Match parameters: channel selection, morphological kernel size, PCA components, k value
3. Compare segmentation cell counts and cluster distributions
4. Verify PCA explained variance ratios match reported values
5. Check that recommended ROIs align with paper's figures
