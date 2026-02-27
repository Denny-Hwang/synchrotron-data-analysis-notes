# Real-Time Data Streaming

## Overview

Once frames leave the detector and pass through the Area Detector framework,
they must be transported to processing nodes with minimal latency. The eBERlight
pipeline employs three complementary streaming technologies -- ZeroMQ (ZMQ),
PV Access, and Globus -- each suited to different latency and bandwidth regimes.

## Streaming Architecture

The general streaming pattern follows a **producer-broker-consumer** model:

```
Producer(s)              Broker / Router           Consumer(s)
+-----------+          +-----------------+       +-------------+
| Area Det  |--ZMQ---->| Stream Router   |------>| Preprocessor|
| IOC       |--PVA---->| (edge node)     |------>| Live Viewer |
+-----------+          +-----------------+       +-------------+
                              |
                        Globus Transfer
                              |
                              v
                       +-------------+
                       | ALCF Compute|
                       | (Polaris /  |
                       |  Aurora)    |
                       +-------------+
```

### Design Principles

1. **Fan-out** -- A single data source feeds multiple consumers simultaneously
   (e.g., live display, disk writer, real-time processing).
2. **Back-pressure** -- Slow consumers do not block the detector; the broker
   drops or queues frames according to configured policy.
3. **Zero-copy** -- ZMQ and shared-memory transports avoid unnecessary copies on
   the local node.

## ZeroMQ (ZMQ) Streaming

### Architecture

ZMQ provides the primary low-latency path for frame-level streaming within the
beamline network.

| Component | ZMQ Pattern | Port | Purpose |
|---|---|---|---|
| Area Detector plugin | PUB | 5555 | Publishes raw frames |
| Stream Router | XSUB / XPUB | 5556 / 5557 | Fans out to consumers |
| Preprocessor | SUB | -- | Subscribes to frame topic |
| Live viewer | SUB | -- | Subscribes for display |

### Message Format

Each ZMQ message is a multi-part envelope:

```
Part 0:  Topic string     ("detector.eiger.frame")
Part 1:  Header (JSON)    {"frame_id": 42, "timestamp": "...", "shape": [2070,2167], "dtype": "uint16"}
Part 2:  Data blob         (raw pixel bytes, optionally LZ4-compressed)
Part 3:  Metadata (JSON)   {"energy_keV": 12.0, "angle_deg": 45.3, ...}
```

### Performance

- **Latency**: < 1 ms for intra-rack transfers (56 Gbps InfiniBand)
- **Throughput**: Sustained 8 GB/s per ZMQ PUB socket (10 x 100 GbE bonded)
- **Reliability**: At-most-once delivery; frame loss rate < 0.01% under load

## PV Access Streaming

PV Access (pvAccess) is the EPICS v7 network protocol, providing structured
data transport with intrinsic type safety and monitoring semantics.

### Use Cases

- **NTNDArray**: Transmits detector frames as Normative Type ND arrays, enabling
  EPICS-native consumers (e.g., CSS displays, archiver appliance).
- **NTScalar / NTTable**: Streams scalar and tabular metadata (motor positions,
  beam intensity) alongside image data.

### Configuration

```yaml
pva_gateway:
  listen_interface: "10.0.0.0/16"      # beamline subnet
  server_addresses:
    - "ioc-det-eiger:5075"
    - "ioc-det-jungfrau:5075"
  access_security: "pva_access.acf"
  max_array_bytes: 50000000             # 50 MB limit per PV
```

### Limitations

PV Access is optimized for control-system semantics (monitors, puts, RPCs)
rather than bulk data transport. For sustained multi-GB/s throughput, ZMQ is
preferred; PV Access handles metadata and lower-rate image streams.

## Globus Data Transfer

### APS to ALCF Connection

Bulk data transfer between the Advanced Photon Source (APS) and the Argonne
Leadership Computing Facility (ALCF) uses Globus with dedicated DTN nodes.

```
APS Beamline Storage (GPFS)
    |
    v
APS Data Transfer Node (DTN)
    |--- 100 Gbps ESnet link ---
    v
ALCF Data Transfer Node (DTN)
    |
    v
ALCF Eagle / Grand filesystem
```

### Transfer Configuration

| Parameter | Value |
|---|---|
| Protocol | GridFTP (Globus Connect Server v5.4) |
| Parallelism | 8 concurrent TCP streams per transfer |
| Concurrency | 4 simultaneous file transfers |
| Encryption | TLS 1.3 for control channel; optional for data |
| Checksum | SHA-256 verified post-transfer |
| Automation | Globus Flows triggered by EPICS scan-complete PV |

### Automated Trigger Flow

1. Scan completes -- EPICS IOC sets `$(P):ScanComplete` PV to 1.
2. Bluesky RunEngine `post_scan_hook` fires a Globus Flow via REST API.
3. Globus Flow stages: transfer data, verify checksum, trigger ALCF job.
4. ALCF batch scheduler (PBS Pro) launches preprocessing container.

### Throughput

- **Measured**: 40-60 Gbps sustained between APS and ALCF DTNs
- **Latency**: Transfer initiation < 5 seconds via Globus Flows
- **Typical scan**: 50 GB tomography dataset transferred in ~8 seconds

## Monitoring and Observability

All streaming components emit metrics to a centralized monitoring stack:

| Metric | Source | Dashboard |
|---|---|---|
| Frame rate (fps) | ZMQ publisher | Grafana: Detector Throughput |
| Transfer rate (Gbps) | Globus activity log | Grafana: Data Movers |
| Consumer lag (frames) | ZMQ broker | Grafana: Stream Health |
| PV update rate (Hz) | pvAccess gateway | Grafana: EPICS Metrics |

Alerts fire when consumer lag exceeds 100 frames or transfer rate drops below
10 Gbps, indicating a bottleneck that may impact real-time processing.

## Next Stage

Streamed data arrives at processing nodes and enters the reconstruction
pipeline described in [processing.md](processing.md).
