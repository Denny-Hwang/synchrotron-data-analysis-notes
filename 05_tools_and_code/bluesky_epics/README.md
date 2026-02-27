# Bluesky / EPICS -- Experiment Orchestration Overview

## What is Bluesky?

Bluesky is a Python-based experiment orchestration framework originally
developed at NSLS-II (Brookhaven National Laboratory) and now adopted across
multiple synchrotron facilities, including APS. It provides a
hardware-abstracted, event-driven system for defining, executing, and
recording experimental procedures.

## What is EPICS?

EPICS (Experimental Physics and Industrial Control System) is the distributed
control system used at APS and most synchrotron facilities worldwide. It
provides real-time communication between hardware devices (motors, detectors,
shutters) via Channel Access (CA) and pvAccess (PVA) network protocols.

## How They Work Together

```
Bluesky RunEngine  (experiment logic, Python)
       |
       v
     ophyd          (hardware abstraction layer)
       |
       v
  EPICS IOCs        (input/output controllers, C)
       |
       v
  Hardware          (motors, detectors, shutters)
```

Bluesky uses `ophyd` device objects that map to EPICS Process Variables (PVs).
When the RunEngine executes a scan plan, ophyd translates high-level commands
(e.g., "move motor to 45 degrees") into EPICS `caput` / `caget` calls.

## Key Components

| Component | Role |
|-----------|------|
| **RunEngine** | Executes scan plans, emits document stream |
| **ophyd** | Python device abstraction over EPICS PVs |
| **Databroker** | Stores and retrieves experimental documents |
| **Bluesky Queue Server** | Remote job queue for unattended operation |
| **Tiled** | Data access service (REST API + Python client) |

## Relevance to APS BER Program

- APS-U beamlines are adopting Bluesky/EPICS as the standard control stack.
- Automated XRF scanning and tomographic data collection can be defined as
  Bluesky plans.
- The ROI Finder can feed recommended ROIs back into Bluesky to drive
  adaptive, closed-loop experiments.

## Related Documents

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Detailed architecture and document model |
