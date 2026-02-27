# System Architecture Diagrams

## Overview

Comprehensive Mermaid-based diagrams of the eBERlight data pipeline showing
component interactions, data format conversions, and system interfaces.

## Full System Architecture

```mermaid
flowchart TB
    subgraph BL["Beamline (APS)"]
        DET[Detector] -->|"raw frames"| IOC[EPICS IOC]
        IOC -->|"HDF5/SWMR"| GPFS[(Beamline GPFS)]
    end
    subgraph EDGE["Edge Computing"]
        IOC -->|"NDArray"| ZMQ_PUB[ZMQ Publisher]
        IOC -->|"NTNDArray"| PVA[PV Access GW]
    end
    subgraph ALCF["ALCF (Polaris / Aurora)"]
        EAGLE[(Eagle FS)]
        PRE[Preprocessing] -->|"HDF5"| REC[TomocuPy]
        REC -->|"Zarr"| DEN[Denoising]
        DEN -->|"Zarr"| SEG[U-Net Segmentation]
        SEG -->|"HDF5"| QNT[Quantification]
        GPFS -->|"Globus 100Gbps"| EAGLE --> PRE
        ZMQ_PUB -->|"ZMQ TCP"| PRE
    end
    subgraph ANALYSIS["Analysis"]
        QNT --> TRT[Triton Inference]
        EAGLE --> JH[JupyterHub]
    end
    subgraph STORE["Storage"]
        EAGLE -->|"Globus Flow"| PET[(Petrel)]
        EAGLE -->|"Globus Flow"| TAPE[(HPSS Tape)]
        PET --> DOI[DataCite DOI]
    end
```

## Data Format Conversion Points

```mermaid
flowchart LR
    A["Vendor Binary"] -->|"IOC decode"| B["HDF5/SWMR"]
    B -->|"NDPluginCodec"| C["ZMQ Multipart"]
    C -->|"consumer"| D["HDF5 float32"]
    B -->|"Globus copy"| D
    D -->|"TomocuPy"| E["Zarr float32"]
    E -->|"DNN"| F["Zarr float32"]
    F -->|"U-Net"| G["HDF5 uint8"]
    G -->|"measure"| H["JSON metrics"]
    D & E & F & G & H -->|"NeXus writer"| I["NeXus/HDF5 archive"]
```

## Interface Protocol Map

```mermaid
flowchart TB
    QS["Queue Server"] -->|"HTTP/JSON"| BS["Bluesky RE"]
    BS -->|"Channel Access"| EP["EPICS IOC"]
    EP -->|"plugin chain"| AD["areaDetector"]
    BS -->|"msgpack"| DB["Databroker"]
    AD -->|"ZMQ PUB"| ZM["ZMQ Broker"]
    BS -->|"REST API"| GF["Globus Flows"]
    GF -->|"GridFTP"| PT["Petrel"]
    GF -->|"PBS qsub"| TC["TomocuPy"]
    TC -->|"Zarr"| PY["PyTorch"]
    PY -->|"HTTP"| ML["MLflow"]
    PT -->|"REST"| DC["DataCite"]
```

## Scan Lifecycle Sequence

```mermaid
sequenceDiagram
    participant User
    participant QS as Queue Server
    participant BS as Bluesky RE
    participant IOC as EPICS IOC
    participant Det as Detector
    participant GF as Globus Flows
    participant HPC as ALCF Compute
    participant Pet as Petrel

    User->>QS: Submit scan plan
    QS->>BS: Execute plan
    BS->>IOC: Configure detector
    IOC->>Det: Start acquisition
    loop Each projection
        Det->>IOC: Frame (trigger)
        IOC->>IOC: Write HDF5 + stream ZMQ
    end
    IOC->>BS: Scan done
    BS->>GF: Trigger transfer
    GF->>HPC: Transfer + process
    GF->>Pet: Archive results
    GF->>User: Notification
```

## Network Topology

```mermaid
flowchart LR
    subgraph APS["APS Network"]
        SW1[Beamline Switch] --- DTN1[APS DTN]
    end
    subgraph ES["ESnet Backbone"]
        RTR[Router 400GbE]
    end
    subgraph ALCF["ALCF Network"]
        DTN2[ALCF DTN] --- SW2[Core Switch]
        SW2 --- GPU[Compute Nodes]
        SW2 --- EGL[Eagle Storage]
    end
    DTN1 ---|"100 Gbps"| RTR ---|"100 Gbps"| DTN2
```

## Legend

| Symbol | Meaning |
|---|---|
| Rectangle | Compute service or application |
| Cylinder | Storage system (filesystem, database, archive) |
| Arrow label | Protocol or data format at the interface |
| Subgraph | Network or logical boundary |

## Related Documents

- [README.md](README.md) -- Pipeline overview
- [acquisition.md](acquisition.md) -- Detector and IOC details
- [streaming.md](streaming.md) -- Transport protocols
- [processing.md](processing.md) -- Reconstruction and ML pipeline
- [analysis.md](analysis.md) -- Inference and visualization
- [storage.md](storage.md) -- Archival and DOI workflow
