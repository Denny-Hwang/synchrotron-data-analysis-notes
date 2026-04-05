# Foundation Models for Beamline Control

**References**: Nature Scientific Data (2025), DOI: [10.1038/s41597-025-04605-9](https://doi.org/10.1038/s41597-025-04605-9); tomoCAM, J. Synchrotron Rad. (2024), DOI: [10.1107/S1600577523009876](https://doi.org/10.1107/S1600577523009876)

## Concept

**Foundation models** are large-scale pre-trained models (LLMs, vision transformers,
multi-modal models) applied to synchrotron beamline operation. Instead of training
task-specific models from scratch for each beamline function, foundation models
leverage massive pre-training to generalize across tasks with zero-shot or few-shot
adaptation.

```
Traditional ML pipeline:
  Beamline task A → Collect data A → Train model A
  Beamline task B → Collect data B → Train model B
  Beamline task C → Collect data C → Train model C

Foundation model approach:
  Large pre-trained model → Fine-tune/prompt for task A, B, C
                          → Zero-shot generalization to new tasks
```

## Architecture

### LLM-Based Beamline Interface

```
User natural language input
    │  "Set the energy to 12 keV and collect 720 projections
    │   with 0.5 second exposure"
    │
    ├─→ LLM (GPT-4 / Claude) with beamline context
    │       System prompt: beamline capabilities, safety limits,
    │                      EPICS PV names, standard procedures
    │
    ├─→ Structured command generation
    │       {
    │         "action": "configure_scan",
    │         "parameters": {
    │           "energy_keV": 12.0,
    │           "n_projections": 720,
    │           "exposure_s": 0.5
    │         }
    │       }
    │
    ├─→ Safety validation layer
    │       Check: energy within range? exposure safe? dose limit?
    │
    └─→ EPICS/Bluesky execution
        caput("energy", 12.0)
        RE(ct_scan(n_proj=720, exp=0.5))
```

### Vision Transformer for Sample Screening

```
Sample microscope image (H×W×3)
    │
    ├─→ Patch embedding (16×16 patches)
    │       Each patch → linear projection → token
    │       + positional embedding + [CLS] token
    │
    ├─→ Transformer encoder (12 layers)
    │       Multi-head self-attention (12 heads)
    │       Layer norm + MLP + residual connections
    │       Pre-trained on ImageNet + fine-tuned on synchrotron data
    │
    ├─→ Task heads (multi-task):
    │       [CLS] → Sample type classification
    │       [CLS] → Quality score regression
    │       Patch tokens → ROI segmentation
    │       [CLS] → Recommended scan parameters
    │
    └─→ Outputs:
        - Sample type: "biological tissue, hydrated"
        - Quality: 0.87 (suitable for measurement)
        - ROI mask: highlighted regions of interest
        - Parameters: {energy: 10 keV, resolution: 0.5 µm}
```

### Multi-Modal Foundation Model

```
Inputs (any combination):
    │
    ├── Optical microscope image ──→ Vision encoder (ViT)
    ├── Previous X-ray data ───────→ Spectral encoder (1D CNN)
    ├── Sample metadata ───────────→ Text encoder (BERT)
    ├── Experimental log ──────────→ Text encoder (BERT)
    │
    ├─→ Cross-attention fusion
    │       All modalities attend to each other
    │       Learns correlations: optical appearance ↔ X-ray properties
    │
    ├─→ Unified representation
    │
    └─→ Task-specific heads:
        ├── Predict optimal beamline parameters
        ├── Estimate expected data quality
        ├── Suggest measurement strategy
        └── Flag potential issues (sample damage, alignment)
```

## Key Capabilities

### Natural Language Experiment Control

```python
# LLM-based beamline assistant example
class BeamlineAssistant:
    def __init__(self, llm_client, beamline_config):
        self.llm = llm_client
        self.config = beamline_config
        self.system_prompt = f"""
        You are a beamline control assistant for {beamline_config['name']}.
        Available motors: {beamline_config['motors']}
        Energy range: {beamline_config['energy_range']} keV
        Safety limits: {beamline_config['safety']}
        Available scan types: {beamline_config['scan_types']}

        Convert user requests into structured commands.
        Always validate against safety limits before execution.
        """

    def process_request(self, user_input):
        response = self.llm.generate(
            system=self.system_prompt,
            user=user_input,
            output_format="json"
        )
        command = self.validate_safety(response)
        return command
```

### Automated Sample Screening

```
Workflow:
  1. Robot loads sample onto stage
  2. Optical microscope captures overview image
  3. Vision transformer classifies sample:
     - Type (biological, geological, materials, ...)
     - Condition (intact, damaged, dried, ...)
     - Regions of interest (edges, inclusions, interfaces)
  4. Foundation model recommends:
     - Scan type (tomo, SAXS, XRF, ...)
     - Energy (K-edge selection for elements of interest)
     - Resolution (feature size dependent)
     - Dose budget (radiation sensitivity)
  5. Beamline executes recommended scan autonomously

Zero-shot capability: Can handle new sample types not seen during training
by leveraging pre-trained visual understanding.
```

### Multi-Task Foundation Models for Synchrotron Data

Trained on diverse synchrotron datasets (CT, diffraction, spectroscopy,
scattering), these models can:

```
Pre-training tasks:
  ├── Masked image modeling (reconstruct masked patches)
  ├── Contrastive learning (match related measurements)
  ├── Next-measurement prediction (temporal context)
  └── Cross-modal alignment (optical ↔ X-ray correspondence)

Downstream tasks (fine-tuned or zero-shot):
  ├── Anomaly detection across modalities
  ├── Phase identification from diffraction
  ├── Segmentation of tomographic volumes
  ├── Spectral decomposition of XRF maps
  └── Quality assessment of incoming data
```

## Three-Dimensional Multimodal Synchrotron Datasets

The 2025 Nature Scientific Data publication provides standardized 3D multimodal
synchrotron datasets for training and benchmarking foundation models:

```
Dataset contents:
  ├── Tomographic volumes (absorption, phase contrast)
  ├── XRF elemental maps (co-registered)
  ├── SAXS/WAXS patterns (spatially resolved)
  ├── Sample metadata and experimental parameters
  └── Annotations and segmentation labels

Use cases for foundation models:
  - Pre-training on diverse synchrotron data
  - Benchmarking cross-modal transfer learning
  - Evaluating zero-shot generalization
```

## tomoCAM Integration

tomoCAM (2024) provides fast model-based iterative reconstruction that can
serve as a computational backbone for foundation model-driven experiments:

```
Foundation model decides: "Collect 180 sparse projections at 15 keV"
    │
    tomoCAM executes: Fast GPU reconstruction in <1 second
    │
    Foundation model evaluates: "Quality sufficient for segmentation task"
    │
    Decision: "Proceed to next sample" or "Collect additional angles"

tomoCAM enables the fast feedback loop required for autonomous operation.
```

## Autonomy Levels with Foundation Models

| Level | Traditional ML | Foundation Model |
|-------|---------------|-----------------|
| **1 - Advisory** | Specialized detector | LLM explains findings in natural language |
| **2 - Supervised** | Task-specific model | Multi-task model handles diverse decisions |
| **3 - Conditional** | Multiple specialized models | Single model covers full workflow |
| **4 - Full autonomous** | Complex pipeline of models | Foundation model orchestrates everything |

## Strengths

1. **Zero-shot generalization**: Handle new sample types without retraining
2. **Natural language interface**: Lowers barrier for non-expert beamline users
3. **Multi-task capability**: Single model handles classification, segmentation, optimization
4. **Transfer learning**: Pre-trained knowledge transfers across beamlines and modalities
5. **Rapid adaptation**: Few-shot fine-tuning for beamline-specific tasks
6. **Interpretability**: LLMs can explain their decisions in natural language

## Limitations

1. **Hallucination risk**: LLMs may generate plausible but incorrect commands
2. **Safety concerns**: Must have robust validation layer before executing commands
3. **Computational cost**: Large models require significant GPU resources
4. **Data requirements**: Pre-training needs large, diverse synchrotron datasets
5. **Latency**: Large model inference may be too slow for real-time control
6. **Domain gap**: General pre-training may not capture synchrotron-specific physics
7. **Reproducibility**: LLM outputs can vary between runs (temperature-dependent)

## Safety Architecture

```
User request
    │
    ├─→ LLM generates command
    │
    ├─→ Layer 1: Schema validation
    │       Is the command well-formed?
    │
    ├─→ Layer 2: Parameter bounds checking
    │       Are all values within safe ranges?
    │
    ├─→ Layer 3: Physics consistency
    │       Is the combination of parameters physically meaningful?
    │
    ├─→ Layer 4: Human approval (for Level 1-2 autonomy)
    │       Display proposed action, wait for confirmation
    │
    └─→ Execute via EPICS/Bluesky

Critical: NEVER allow LLM to directly control hardware without validation.
```

## Code Example

```python
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class SynchrotronVisionModel(nn.Module):
    """Vision transformer for automated sample screening at synchrotron beamlines."""

    def __init__(self, n_sample_types=20, n_scan_params=8):
        super().__init__()
        # Pre-trained ViT backbone
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
        self.backbone = ViTModel(config)

        # Task heads
        self.sample_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_sample_types),
        )

        self.quality_scorer = nn.Sequential(
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.param_predictor = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, n_scan_params),
        )

    def forward(self, images):
        """Predict sample type, quality, and recommended parameters."""
        outputs = self.backbone(images)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token

        sample_type = self.sample_classifier(cls_token)
        quality = self.quality_scorer(cls_token)
        params = self.param_predictor(cls_token)

        return {
            'sample_type': sample_type,      # (batch, n_types) logits
            'quality_score': quality,         # (batch, 1) in [0, 1]
            'scan_parameters': params,        # (batch, n_params) predicted values
        }


class BeamlineLLMController:
    """LLM-based natural language interface for beamline control."""

    def __init__(self, llm_client, beamline_name="2-BM"):
        self.llm = llm_client
        self.beamline = beamline_name
        self.safety_limits = {
            "energy_keV": (5.0, 40.0),
            "exposure_s": (0.001, 10.0),
            "n_projections": (1, 10000),
            "sample_x_mm": (-50.0, 50.0),
            "sample_y_mm": (-25.0, 25.0),
        }

    def parse_command(self, natural_language_request):
        """Convert natural language to structured beamline command."""
        prompt = f"""
        Beamline: {self.beamline}
        Request: {natural_language_request}

        Generate a JSON command with fields:
        - action: one of [move, scan, configure, query]
        - parameters: dict of parameter names and values
        - estimated_time_s: expected execution time
        """
        response = self.llm.generate(prompt)
        command = self._parse_json(response)
        return command

    def validate_command(self, command):
        """Check command against safety limits."""
        errors = []
        for param, value in command.get("parameters", {}).items():
            if param in self.safety_limits:
                lo, hi = self.safety_limits[param]
                if not (lo <= value <= hi):
                    errors.append(
                        f"{param}={value} outside safe range [{lo}, {hi}]"
                    )
        return len(errors) == 0, errors

    def execute(self, natural_language_request):
        """Full pipeline: parse → validate → execute."""
        command = self.parse_command(natural_language_request)
        is_safe, errors = self.validate_command(command)

        if not is_safe:
            return {"status": "rejected", "errors": errors}

        # Execute via EPICS/Bluesky (placeholder)
        result = self._execute_epics(command)
        return {"status": "executed", "result": result}
```

## Relevance to APS BER Program

### Key Applications

- **Automated BER sample screening**: Vision models classify soil, root, and biofilm
  samples; recommend imaging parameters based on sample type
- **Natural language experiment logs**: LLMs parse and structure experimental notes,
  enabling searchable, machine-readable metadata
- **Cross-beamline transfer**: Foundation model trained at 2-BM transfers to 26-ID
  with minimal fine-tuning
- **Real-time decision support**: Multi-modal model advises operators on measurement
  strategy based on incoming data quality

### Beamline Integration

- **2-BM**: Automated tomography screening with vision transformer sample classification
- **26-ID**: Intelligent ptychography scanning guided by real-time quality assessment
- **9-ID**: SAXS/WAXS experiment planning via natural language interface
- **All beamlines**: LLM-based log parsing and metadata extraction for FAIR data practices
- Computational support from ALCF for hosting and running large foundation models

### Potential for Zero-Shot and Few-Shot Generalization

```
Scenario: New beamline commissioning
Traditional: Collect months of training data → Train specialized models
Foundation:  Deploy pre-trained model → Few-shot adapt with 10-50 examples
             or zero-shot with detailed prompt engineering

Scenario: Novel sample type (never measured before)
Traditional: No ML assistance available
Foundation:  Vision model leverages general visual understanding
             LLM leverages scientific knowledge from pre-training
             → Provides reasonable initial parameters

This dramatically reduces the barrier to deploying ML at new beamlines
and for new experimental campaigns.
```

## References

1. "Three-dimensional, multimodal synchrotron data for machine learning applications."
   Nature Scientific Data, 2025. DOI: [10.1038/s41597-025-04605-9](https://doi.org/10.1038/s41597-025-04605-9)
2. Nikitin, V., et al. "tomoCAM: fast model-based iterative reconstruction."
   J. Synchrotron Rad., 2024. DOI: [10.1107/S1600577523009876](https://doi.org/10.1107/S1600577523009876)
3. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image
   Recognition at Scale." ICLR 2021. arXiv: 2010.11929
4. Brown, T., et al. "Language Models are Few-Shot Learners." NeurIPS 2020.
   DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
