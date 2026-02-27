# System Architecture Diagrams

## Overview

This document provides comprehensive Mermaid-based diagrams of the eBERlight
data pipeline architecture, focusing on component interactions, data format
conversion points, and interfaces between systems.

## Full System Architecture

```mermaid
flowchart TB
    subgraph BL["Beamline (APS Sector)"]
        DET[Detector<br/>EIGER / Jungfrau / Vortex]
        IOC[EPICS IOC<br/>areaDetector]
        BL_GPFS[(Beamline<br/>GPFS Storage)]
        DET -->|"raw frames<br/>(vendor binary)"| IOC
        IOC -->|"HDF5/SWMR"| BL_GPFS
    end

    subgraph EDGE["Edge Computing (Beamline Rack)"]
        ZMQ_PUB[ZMQ Publisher]
        PVA_GW[PV Access Gateway]
        LIVE[Live Viewer<br/>CSS / PyDM]
        IOC -->|"NDArray plugin<br/>(shared memory)"| ZMQ_PUB
        IOC -->|"NTNDArray<br/>(pvAccess)"| PVA_GW
        PVA_GW --> LIVE
    end

    subgraph NET["Network Layer"]
        GLB_DTN_APS[Globus DTN<br/>APS]
        GLB_DTN_ALCF[Globus DTN<br/>ALCF]
        BL_GPFS -->|"GridFTP<br/>100 Gbps"| GLB_DTN_APS
        GLB_DTN_APS -->|"ESnet<br/>100 Gbps"| GLB_DTN_ALCF
    end

    subgraph ALCF["ALCF (Polaris / Aurora)"]
        EAGLE[(Eagle<br/>Filesystem)]
        PREPROC[Preprocessing<br/>dark/flat/norm]
        RECON[Reconstruction<br/>TomocuPy]
        DENOISE[Denoising<br/>DNN Models]
        SEGMENT[Segmentation<br/>U-Net]
        QUANT[Quantification]
        GLB_DTN_ALCF --> EAGLE
        ZMQ_PUB -->|"ZMQ TCP<br/>(raw frames)"| PREPROC
        EAGLE --> PREPROC
        PREPROC -->|"float32 HDF5"| RECON
        RECON -->|"float32 Zarr"| DENOISE
        DENOISE -->|"float32 Zarr"| SEGMENT
        SEGMENT -->|"uint8 HDF5<br/>(labels)"| QUANT
    end

    subgraph ANALYSIS["Analysis Services"]
        MLFLOW[MLflow<br/>Model Registry]
        TRITON[Triton<br/>Inference Server]
        JHUB[JupyterHub]
        STLIT[Streamlit<br/>Dashboards]
        NAPARI[Napari<br/>Viewer]
        QUANT --> TRITON
        MLFLOW -->|"model artifacts"| TRITON
        EAGLE --> JHUB
        EAGLE --> STLIT
        EAGLE --> NAPARI
    end

    subgraph STORE["Long-Term Storage"]
        PETREL[(Petrel<br/>Object Store)]
        TAPE[(HPSS<br/>Tape Archive)]
        DOI_SVC[DOI Service<br/>DataCite]
        EAGLE -->|"Globus Flow"| PETREL
        EAGLE -->|"Globus Flow"| TAPE
        PETREL --> DOI_SVC
    end
```

## Data Format Conversion Points

The pipeline performs several format transformations as data moves through
stages. The following diagram highlights each conversion.

```mermaid
flowchart LR
    A["Vendor Binary<br/>(detector native)"]
    B["HDF5 / SWMR<br/>(areaDetector)"]
    C["ZMQ Multipart<br/>(raw bytes + JSON header)"]
    D["HDF5<br/>(preprocessed float32)"]
    E["Zarr<br/>(reconstructed float32)"]
    F["Zarr<br/>(denoised float32)"]
    G["HDF5<br/>(segmentation uint8)"]
    H["JSON<br/>(quantification metrics)"]
    I["NeXus / HDF5<br/>(archive master file)"]

    A -->|"IOC device support<br/>decode + reshape"| B
    B -->|"NDPluginCodec<br/>serialize + compress"| C
    C -->|"ZMQ consumer<br/>deserialize"| D
    B -->|"Globus transfer<br/>file copy"| D
    D -->|"TomocuPy<br/>reconstruct"| E
    E -->|"DNN inference<br/>denoise"| F
    F -->|"U-Net inference<br/>segment"| G
    G -->|"morphometrics<br/>measure"| H
    D & E & F & G & H -->|"NeXus writer<br/>assemble"| I
```

## Interface Map

Each arrow below represents a distinct interface protocol or API between
eBERlight subsystems.

```mermaid
flowchart TB
    EPICS["EPICS IOC"]
    BLUESKY["Bluesky<br/>RunEngine"]
    QSERVER["Queue Server"]
    AD["areaDetector"]
    DATABROKER["Databroker<br/>(MongoDB)"]
    ZMQ["ZMQ Broker"]
    GLOBUS_FLOWS["Globus Flows"]
    TOMOCUPY["TomocuPy"]
    PYTORCH["PyTorch<br/>Models"]
    MLFLOW["MLflow"]
    TRITON["Triton Server"]
    PETREL["Petrel"]
    HPSS["HPSS Tape"]
    DATACITE["DataCite API"]

    QSERVER -->|"RE commands<br/>(HTTP/JSON)"| BLUESKY
    BLUESKY -->|"ophyd devices<br/>(Channel Access)"| EPICS
    EPICS -->|"plugin chain<br/>(internal)"| AD
    BLUESKY -->|"event documents<br/>(dict/msgpack)"| DATABROKER
    AD -->|"NDArray<br/>(ZMQ PUB)"| ZMQ
    BLUESKY -->|"REST API<br/>(scan complete)"| GLOBUS_FLOWS
    GLOBUS_FLOWS -->|"transfer task<br/>(GridFTP)"| PETREL
    GLOBUS_FLOWS -->|"transfer task<br/>(GridFTP)"| HPSS
    GLOBUS_FLOWS -->|"PBS qsub<br/>(SSH)"| TOMOCUPY
    TOMOCUPY -->|"Zarr output<br/>(filesystem)"| PYTORCH
    MLFLOW -->|"model URI<br/>(HTTP)"| TRITON
    PYTORCH -->|"log metrics<br/>(HTTP)"| MLFLOW
    TRITON -->|"gRPC / HTTP<br/>(inference)"| PETREL
    PETREL -->|"metadata<br/>(REST API)"| DATACITE
```

## Component Interaction: Scan Lifecycle

This sequence diagram shows the interaction between components during a
single tomography scan.

```mermaid
sequenceDiagram
    participant User
    participant QServer as Queue Server
    participant Bluesky as Bluesky RE
    participant EPICS as EPICS IOC
    participant Det as Detector
    participant ZMQ as ZMQ Broker
    participant Globus as Globus Flows
    participant ALCF as ALCF Compute
    participant Petrel as Petrel

    User->>QServer: Submit scan plan
    QServer->>Bluesky: Execute plan
    Bluesky->>EPICS: Configure detector (caput)
    Bluesky->>EPICS: Arm motor triggers
    EPICS->>Det: Start acquisition
    loop Each projection
        Det->>EPICS: Frame (hardware trigger)
        EPICS->>ZMQ: Publish frame (NDArray)
        EPICS->>EPICS: Write HDF5 (SWMR)
    end
    Det->>EPICS: Acquisition complete
    EPICS->>Bluesky: Scan done (CA monitor)
    Bluesky->>Globus: Trigger transfer flow
    Globus->>ALCF: Transfer dataset
    ALCF->>ALCF: Preprocess + Reconstruct
    ALCF->>ALCF: Denoise + Segment
    ALCF->>Globus: Processing complete
    Globus->>Petrel: Archive results
    Globus->>User: Email notification
```

## Network Topology

```mermaid
flowchart LR
    subgraph APS_NET["APS Network (10.0.x.x)"]
        BL_SW[Beamline Switch<br/>100 GbE]
        IOC_SRV[IOC Server]
        DTN_APS[DTN Node]
        BL_SW --- IOC_SRV
        BL_SW --- DTN_APS
    end

    subgraph ESNET["ESnet Backbone"]
        ESNET_RTR[ESnet Router<br/>400 GbE]
    end

    subgraph ALCF_NET["ALCF Network (10.1.x.x)"]
        ALCF_SW[Core Switch<br/>400 GbE]
        DTN_ALCF[DTN Node]
        COMPUTE[Polaris / Aurora<br/>Compute Nodes]
        EAGLE_FS[Eagle Storage]
        ALCF_SW --- DTN_ALCF
        ALCF_SW --- COMPUTE
        ALCF_SW --- EAGLE_FS
    end

    DTN_APS ---|"100 Gbps<br/>dedicated"| ESNET_RTR
    ESNET_RTR ---|"100 Gbps<br/>dedicated"| DTN_ALCF
```

## Legend

| Symbol | Meaning |
|---|---|
| Rectangle | Compute service or application |
| Cylinder | Storage system (filesystem, database, archive) |
| Arrow label | Protocol or data format at the interface |
| Subgraph | Network or logical boundary |

## Related Documents

- [README.md](README.md) -- Pipeline overview and stage summary
- [acquisition.md](acquisition.md) -- Detector and IOC details
- [streaming.md](streaming.md) -- Transport protocol configuration
- [processing.md](processing.md) -- Reconstruction and ML pipeline
- [analysis.md](analysis.md) -- Inference and visualization
- [storage.md](storage.md) -- Archival and DOI workflow
