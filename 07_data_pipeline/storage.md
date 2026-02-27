# Data Storage and Management

## Overview

The storage stage encompasses data transfer and replication via Globus,
long-term archival on tape, active storage on Petrel, metadata standardization
with NeXus/HDF5, data sharing mechanisms, and DOI assignment.

## Storage Tiers

| Tier | System | Capacity | Access Latency | Retention |
|---|---|---|---|---|
| Hot | Eagle (GPFS) | 100 TB allocation | Milliseconds | 90 days |
| Warm | Petrel (Object Store) | 500 TB allocation | Seconds | 3 years |
| Cold | HPSS Tape Archive | Unlimited | Minutes-hours | 10+ years |

Data automatically migrates through tiers based on age and access frequency.

## Globus Data Management

Globus serves as the unified data management layer for transfers between
beamline storage, ALCF filesystems, Petrel, and collaborator endpoints.

| Endpoint | Location | Filesystem |
|---|---|---|
| `aps#data` | APS | GPFS |
| `alcf#eagle` | ALCF | Eagle / GPFS |
| `anl#petrel` | Argonne | Object Store |
| `nersc#dtn` | NERSC | CFS / HPSS |

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

Tape archives follow a hierarchical year/scan structure under `/eBERlight/archive/`.
Each scan directory contains `raw/`, `processed/`, `metadata/`, and `analysis/`
subdirectories.

| Data Class | Retention | Copies | Justification |
|---|---|---|---|
| Raw detector data | 10 years | 2 (tape + Petrel) | DOE data management mandate |
| Processed volumes | 5 years | 1 (tape) | Reproducible from raw |
| Analysis results | 5 years | 1 (Petrel) | Reproducible from processed |
| Publication datasets | Permanent | 2 (tape + DOI repo) | Linked to publications |

## Petrel Object Store

Petrel provides a web portal for browsing datasets, Globus Auth for access
control, a REST API via Globus SDK, and a search index for metadata-driven
dataset discovery.

```python
from globus_sdk import TransferClient, AccessTokenAuthorizer

tc = TransferClient(authorizer=AccessTokenAuthorizer(token))
rule = {
    "DATA_TYPE": "access",
    "principal_type": "identity",
    "principal": "collaborator@university.edu",
    "path": "/eBERlight/archive/scan_0042/",
    "permissions": "r"
}
tc.add_endpoint_acl_rule(petrel_endpoint_id, rule)
```

## Metadata Standards (NeXus)

All datasets conform to NeXus application definitions in HDF5 master files:

| Technique | Definition | Key Groups |
|---|---|---|
| Tomography | NXtomo | NXsample, NXdetector, NXsource |
| SAXS/WAXS | NXsas | NXcollimator, NXdetector, NXsample |
| XAS/XANES | NXxas | NXmonochromator, NXdetector |
| XRF mapping | NXfluo | NXdetector, NXsample |

```
/entry (NXentry)
    /instrument (NXinstrument)
        /source: energy=7.0 GeV, current=100.0 mA
        /detector: description="EIGER2 X 4M", exposure_time=0.001 s
    /sample (NXsample): name, temperature
    /data (NXdata): @signal="data"
    /process (NXprocess): program, version, parameters
```

Files are validated at ingest with `cnxvalidate`. Failures are quarantined.

## Data Sharing

| Method | Audience | Access Control |
|---|---|---|
| Globus shared endpoint | Named collaborators | Identity-based ACL |
| Petrel web portal | Broader community | Globus Auth groups |
| Materials Data Facility | Public datasets | Open access after embargo |
| Zenodo / Figshare | Publication supplements | DOI-linked open access |

Datasets are embargoed for **12 months** from collection (or until publication),
then transition to open access under CC-BY 4.0 licensing.

## DOI Assignment

1. PI requests DOI via the eBERlight data portal.
2. Metadata is validated against DataCite schema.
3. DOI is minted through Argonne's DataCite membership.
4. Landing page is created on Petrel with download links.
5. DOI is registered and linked to the dataset.

```json
{
  "doi": "10.18126/eBERlight.scan_0042",
  "creators": [{"name": "Smith, J.", "affiliation": "Argonne"}],
  "title": "In-situ tomography of NMC811 cathode cycling",
  "publisher": "Argonne National Laboratory",
  "publicationYear": 2025,
  "resourceType": "Dataset",
  "rights": "CC-BY 4.0"
}
```

DOI landing pages track download counts and citation metrics for reporting.

## Related Documents

- [acquisition.md](acquisition.md) -- Data origin and detector metadata
- [processing.md](processing.md) -- Transformation provenance
- [analysis.md](analysis.md) -- Validation and QA records stored with data
