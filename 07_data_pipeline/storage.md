# Data Storage and Management

## Overview

The storage stage is the final phase of the eBERlight data pipeline. It
encompasses data transfer and replication via Globus, long-term archival on
tape, active storage on Petrel, metadata standardization with NeXus/HDF5,
data sharing mechanisms, and DOI assignment for publication-ready datasets.

## Storage Tiers

| Tier | System | Capacity | Access Latency | Retention |
|---|---|---|---|---|
| Hot | Eagle (GPFS) | 100 TB allocation | Milliseconds | 90 days |
| Warm | Petrel (Object Store) | 500 TB allocation | Seconds | 3 years |
| Cold | HPSS Tape Archive | Unlimited | Minutes-hours | 10+ years |

Data automatically migrates through tiers based on age and access frequency.

## Globus Data Management

### Transfer Infrastructure

Globus serves as the unified data management layer, handling transfers between
beamline storage, ALCF filesystems, Petrel, and external collaborator
endpoints.

```
APS Beamline GPFS ──> ALCF Eagle (hot processing)
                 └──> Petrel (warm sharing)
                 └──> HPSS Tape (cold archive)
                 └──> Collaborator Endpoints (sharing)
```

### Globus Endpoints

| Endpoint | Location | Type | Filesystem |
|---|---|---|---|
| `aps#data` | APS | Globus Connect Server | GPFS |
| `alcf#eagle` | ALCF | Globus Connect Server | Eagle / GPFS |
| `anl#petrel` | Argonne | Globus Connect Server | Object Store |
| `nersc#dtn` | NERSC | Globus Connect Server | CFS / HPSS |

### Automated Transfer Policies

```yaml
globus_flows:
  post_processing:
    trigger: "processing_complete"
    steps:
      - transfer:
          source: "alcf#eagle:/eBERlight/processed/${SCAN_ID}"
          destination: "anl#petrel:/eBERlight/archive/${SCAN_ID}"
          verify_checksum: true
      - transfer:
          source: "alcf#eagle:/eBERlight/processed/${SCAN_ID}"
          destination: "hpss#tape:/eBERlight/archive/${YEAR}/${SCAN_ID}"
          verify_checksum: true
      - notify:
          email: "${PI_EMAIL}"
          message: "Dataset ${SCAN_ID} archived successfully"
```

## Tape Archive (HPSS)

### Organization

Tape archives follow a hierarchical structure:

```
/eBERlight/
    archive/
        2025/
            scan_0001/
                raw/                 # Original detector frames
                processed/           # Reconstructed volumes
                metadata/            # NeXus master files
                analysis/            # Segmentation, quantification
            scan_0002/
            ...
        2026/
            ...
```

### Retention Policy

| Data Class | Retention | Copies | Justification |
|---|---|---|---|
| Raw detector data | 10 years | 2 (tape + Petrel) | DOE data management mandate |
| Processed volumes | 5 years | 1 (tape) | Reproducible from raw |
| Analysis results | 5 years | 1 (Petrel) | Reproducible from processed |
| Publication datasets | Permanent | 2 (tape + DOI repository) | Linked to publications |

## Petrel Object Store

Petrel is Argonne's research data service built on top of object storage with
Globus-integrated access control.

### Features

- **Web portal** for browsing and downloading datasets
- **Globus Auth** integration for identity-based access control
- **REST API** for programmatic access via Globus SDK
- **Search index** for metadata-driven dataset discovery

### Access Control

```python
from globus_sdk import TransferClient, AccessTokenAuthorizer

tc = TransferClient(authorizer=AccessTokenAuthorizer(token))

# Share dataset with collaborator
rule = {
    "DATA_TYPE": "access",
    "principal_type": "identity",
    "principal": "collaborator@university.edu",
    "path": "/eBERlight/archive/scan_0042/",
    "permissions": "r",
    "notify_email": "collaborator@university.edu"
}
tc.add_endpoint_acl_rule(petrel_endpoint_id, rule)
```

## Metadata Standards (NeXus)

### NeXus Application Definitions

All eBERlight datasets conform to NeXus application definitions stored in
HDF5 master files:

| Technique | Definition | Key Groups |
|---|---|---|
| Tomography | NXtomo | NXsample, NXdetector, NXsource |
| SAXS/WAXS | NXsas | NXcollimator, NXdetector, NXsample |
| XAS/XANES | NXxas | NXmonochromator, NXdetector |
| XRF mapping | NXfluo | NXdetector, NXsample |

### Metadata Schema

```
/entry (NXentry)
    /instrument (NXinstrument)
        /source (NXsource)
            energy = 7.0 [GeV]
            current = 100.0 [mA]
        /detector (NXdetector)
            description = "EIGER2 X 4M"
            exposure_time = 0.001 [s]
    /sample (NXsample)
        name = "battery_cathode_NMC811"
        temperature = 298.15 [K]
    /data (NXdata)
        @signal = "data"
        data -> /entry/instrument/detector/data
    /process (NXprocess)
        program = "TomocuPy"
        version = "1.2.0"
        parameters = {...}
```

### Validation

NeXus files are validated at ingest using `cnxvalidate`:

```bash
cnxvalidate -a NXtomo scan_0042_master.nxs
```

Files that fail validation are quarantined and flagged for manual correction.

## Data Sharing

### Sharing Mechanisms

| Method | Audience | Access Control |
|---|---|---|
| Globus shared endpoint | Named collaborators | Identity-based ACL |
| Petrel web portal | Broader community | Globus Auth groups |
| Materials Data Facility | Public datasets | Open access after embargo |
| Zenodo / Figshare | Publication supplements | DOI-linked open access |

### Embargo Policy

Datasets are embargoed for **12 months** from collection date (or until
publication, whichever comes first). After embargo, datasets transition to
open access with appropriate licensing (CC-BY 4.0).

## DOI Assignment

### Process

1. Principal investigator requests DOI via the eBERlight data portal.
2. Metadata is validated against DataCite schema requirements.
3. DOI is minted through Argonne's DataCite membership.
4. Landing page is created on the Petrel web portal.
5. DOI is registered and linked to the dataset.

### DataCite Metadata

```json
{
  "doi": "10.18126/eBERlight.scan_0042",
  "creators": [{"name": "Smith, J.", "affiliation": "Argonne"}],
  "title": "In-situ tomography of NMC811 cathode cycling",
  "publisher": "Argonne National Laboratory",
  "publicationYear": 2025,
  "resourceType": "Dataset",
  "subjects": ["tomography", "battery", "cathode"],
  "rights": "CC-BY 4.0",
  "relatedIdentifiers": [
    {"identifier": "10.1234/journal.paper.2025", "relationType": "IsSupplementTo"}
  ]
}
```

### DOI Statistics

DOI landing pages track download counts and citation metrics, providing PIs
with usage data for reporting to funding agencies.

## Related Documents

- [acquisition.md](acquisition.md) -- Data origin and detector metadata
- [processing.md](processing.md) -- Transformation provenance
- [analysis.md](analysis.md) -- Validation and QA records stored with data
