# Bluesky / EPICS -- Architecture Details

## RunEngine

The RunEngine is the core execution engine for Bluesky. It consumes a
**plan** (a Python generator that yields **messages**) and translates each
message into hardware actions and document emissions.

### Message Types

| Message | Description |
|---------|-------------|
| `open_run` | Start a new run; emits a RunStart document |
| `set` | Command a device to move to a value |
| `trigger` | Trigger a detector to acquire |
| `read` | Read current values from a device |
| `save` | Bundle current readings into an Event document |
| `close_run` | End the run; emits a RunStop document |

### Execution Flow

```python
from bluesky import RunEngine
from bluesky.plans import scan
from ophyd import EpicsMotor, EpicsSignalRO

motor = EpicsMotor("IOC:m1", name="motor")
det = EpicsSignalRO("IOC:det:total", name="detector")

RE = RunEngine({})
RE(scan([det], motor, 0, 180, 100))
```

The `scan` plan yields a sequence of messages:
`open_run -> (set, trigger, read, save) x N -> close_run`

## ophyd Device Model

ophyd provides a hierarchy of device classes.

| Class | Use |
|-------|-----|
| `Signal` | Single PV (read or read-write) |
| `EpicsSignal` | Signal backed by an EPICS PV |
| `EpicsMotor` | Motor with position, velocity, and status PVs |
| `Device` | Composite grouping of Signals |
| `AreaDetector` | Multi-PV detector with plugin chain |

### Device Configuration vs Read

Each device distinguishes between:

- **read()** -- primary data (e.g., motor position, detector counts).
- **read_configuration()** -- slow-changing metadata (e.g., gain, exposure
  time).

Both are captured in the document stream.

## Document Model

Bluesky emits a stream of JSON-serialisable documents during every run.

| Document | Emitted when | Contents |
|----------|-------------|----------|
| **RunStart** | `open_run` | UID, plan name, metadata, timestamp |
| **Descriptor** | First `save` | Data keys, shapes, dtypes, source PVs |
| **Event** | Each `save` | Timestamped data readings |
| **EventPage** | Bulk `save` | Columnar batch of Events (performance) |
| **Resource** | External file | File path, spec (HDF5, TIFF) |
| **Datum** | External data | Pointer into Resource |
| **RunStop** | `close_run` | Exit status, end timestamp |

### Document Flow

```
RunEngine
   |
   +---> Callback 1: LiveTable (print to terminal)
   +---> Callback 2: LivePlot (matplotlib)
   +---> Callback 3: Databroker insert (MongoDB / msgpack)
   +---> Callback 4: Custom analysis callback
```

## Databroker

Databroker provides post-hoc access to the document stream.

```python
from databroker import catalog

cat = catalog["my_beamline"]
run = cat[-1]           # most recent run
ds = run.primary.read() # xarray Dataset
```

Storage back-ends: MongoDB + GridFS (legacy), msgpack files (Databroker v2),
or Tiled server.

## Bluesky Queue Server

For unattended or remote operation, the Queue Server wraps the RunEngine in
a ZMQ-based service.

- Plans are submitted as JSON descriptions to a queue.
- The server executes plans sequentially (or with configurable concurrency).
- Status and results are accessible via REST API or the `bluesky-widgets` GUI.

## Integration Pattern for APS

1. **Data collection** -- Bluesky plan executes an XRF raster scan via EPICS
   motors and areaDetector.
2. **Data landing** -- HDF5 files written by areaDetector file plugin.
3. **Analysis trigger** -- a Bluesky callback detects RunStop and launches
   the ROI Finder pipeline.
4. **Adaptive feedback** -- ROI Finder results are written to EPICS PVs or
   submitted as a new Bluesky plan targeting the recommended ROIs for
   high-resolution re-scan.
