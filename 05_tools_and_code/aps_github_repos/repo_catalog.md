# APS GitHub Repository Catalogue

Repositories are grouped by category. Each entry lists the repository name,
organisation, language, and a brief description.

---

## 1. Beamline Controls

| Repository | Org | Language | Description |
|-----------|-----|----------|-------------|
| epics-base | epics-base | C | Core EPICS runtime, Channel Access, pvAccess |
| ADCore | areaDetector | C++ | EPICS areaDetector core framework |
| ADSimDetector | areaDetector | C++ | Simulated detector for testing |
| ADPilatus | areaDetector | C++ | Dectris Pilatus detector driver |
| ADEiger | areaDetector | C++ | Dectris Eiger detector driver |
| motor | epics-modules | C | EPICS motor record and device support |
| ioc-deploy-tools | BCDA-APS | Python | IOC deployment and version management |
| aps-dm | aps-anl | Python | APS Data Management system CLI/API |

## 2. Data Analysis

| Repository | Org | Language | Description |
|-----------|-----|----------|-------------|
| tomopy | tomography | Python/C | Tomographic reconstruction library |
| tomocupy | tomography | Python/CUDA | GPU-accelerated tomographic reconstruction |
| dxchange | tomography | Python | Data Exchange HDF5 I/O utilities |
| dxfile | tomography | Python | Data Exchange file specification |
| tomobank | tomography | Python | Tomographic test dataset repository |
| xrf-maps | aps-anl | C++ | Open-source XRF map fitting (alternative to MAPS) |
| PyXRF | NSLS-II | Python | Python XRF fitting and visualization |
| larch | xraypy | Python | XAFS/XRF analysis (Larch) |
| xraylarch | xraypy | Python | X-ray spectroscopy analysis tools |

## 3. Simulation

| Repository | Org | Language | Description |
|-----------|-----|----------|-------------|
| shadow3 | oasys-kit | Fortran/C | Ray-tracing for X-ray optics |
| SRW | ochubar | C++/Python | Synchrotron Radiation Workshop (wave optics) |
| OASYS1 | oasys-kit | Python | GUI integrating Shadow3 and SRW |
| xoppy | oasys-kit | Python | X-ray optics utilities (undulator spectra, etc.) |
| aps-undulator | aps-anl | Python | APS undulator field and spectrum calculations |

## 4. AI / ML

| Repository | Org | Language | Description |
|-----------|-----|----------|-------------|
| mlexchange | mlexchange | Python | ML platform (see mlexchange/ docs) |
| DLSIA | mlexchange | Python | Deep Learning for Scientific Image Analysis |
| tike | tomography | Python | Ptychography and tomography with ML priors |
| pvauto | aps-anl | Python | Automated alignment using ML on EPICS PVs |
| ai-science-training | argonne-lcf | Python | Argonne AI/ML training materials |

## 5. Workflow and Orchestration

| Repository | Org | Language | Description |
|-----------|-----|----------|-------------|
| bluesky | bluesky | Python | Experiment orchestration RunEngine |
| ophyd | bluesky | Python | Hardware abstraction for Bluesky |
| databroker | bluesky | Python | Data access and search for Bluesky documents |
| bluesky-queueserver | bluesky | Python | Remote experiment queue server |
| tiled | bluesky | Python | Data access service (REST + Python client) |
| happi | pcdshds | Python | Hardware database for device configuration |

---

## Notes

- Repositories listed are those most relevant to the BER program scope
  (XRF, tomography, ML, beamline control).
- Some repositories span categories (e.g., `tike` bridges data analysis and
  AI).
- Repository activity and maintenance status should be checked before
  adopting any package.
