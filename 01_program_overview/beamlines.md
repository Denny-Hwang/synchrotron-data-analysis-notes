# eBERlight Beamlines

eBERlight operates **15 beamlines** at the Advanced Photon Source, organized into five
technique categories. Each beamline is optimized for specific experimental modalities
relevant to biological and environmental research.

## Beamline Summary

### Crystallography Beamlines

| Beamline | Name/Designation | Technique | Energy Range | Beam Size |
|----------|-----------------|-----------|-------------|-----------|
| **21-ID-D** | LS-CAT | MX, MAD/SAD | 6.5–20 keV | 20–150 µm |
| **21-ID-F** | LS-CAT | MX, Screening | 12.7 keV (fixed) | 50–300 µm |
| **21-ID-G** | LS-CAT | MX, SSX | 6.5–20 keV | 10–100 µm |

**Detectors**: EIGER 9M/16M (hybrid pixel), PILATUS 6M
**Data Output**: HDF5 (EIGER native), CBF (legacy), ~1–100 GB per dataset
**Research Focus**: Protein structure determination, enzyme mechanisms, macromolecular complexes

#### Key Capabilities
- **MX** (Macromolecular Crystallography): Standard single-crystal diffraction
- **SSX** (Serial Synchrotron Crystallography): Serial data collection from microcrystals
- **MAD/SAD**: Multi/Single-wavelength anomalous dispersion for phase determination
- Post APS-U: microfocus capability down to ~10 µm enables microcrystallography

---

### Imaging Beamlines

| Beamline | Name/Designation | Technique | Energy Range | Resolution |
|----------|-----------------|-----------|-------------|------------|
| **2-BM-A** | XSD-IMG | µCT, Fast Tomography | 10–40 keV | ~1 µm pixel |
| **7-BM-B** | QM2 | µCT, In-situ | 6–80 keV | 0.5–5 µm pixel |
| **32-ID-B/C** | XSD-IMG | Nano-CT, Phase Contrast | 7–40 keV | 50–100 nm |

**Detectors**: Scintillator-coupled cameras (PCO Edge, Grasshopper), area detectors
**Data Output**: HDF5, TIFF stacks; projections 2048×2048 to 4096×4096 pixels; 10–500 GB/scan
**Research Focus**: Soil microstructure, root architecture, porous media, in-situ reactions

#### Key Capabilities
- **2-BM-A**: High-throughput tomography (sub-second scans possible), environmental cells
- **7-BM-B**: Broad energy range for dual-energy imaging, in-situ capabilities
- **32-ID-B/C**: Transmission X-ray microscopy (TXM), Fresnel zone plate optics, highest spatial resolution

---

### Microscopy Beamlines

| Beamline | Name/Designation | Technique | Energy Range | Beam Size |
|----------|-----------------|-----------|-------------|-----------|
| **2-ID-D** | XSD-MIC | XRF Nanoprobe | 5–30 keV | 30–200 nm |
| **2-ID-E** | XSD-MIC | XRF Microprobe, Ptychography | 5–30 keV | 0.5–5 µm |
| **8-BM-B** | XSD-MIC | XRF, XANES Imaging | 4.5–20 keV | 2–20 µm |
| **33-ID-C** | XSD-MIC | Ptychography, CDI | 6–25 keV | Coherent beam |

**Detectors**: Vortex ME-4 (energy-dispersive, XRF), Eiger 500K (ptychography)
**Data Output**: HDF5 multi-channel elemental maps, diffraction patterns; 1–100 GB/scan
**Research Focus**: Elemental distribution in cells, nutrient uptake, contaminant mapping

#### Key Capabilities
- **2-ID-D**: Nanoscale elemental mapping, sub-100 nm resolution, trace element sensitivity
- **2-ID-E**: Combined XRF + ptychography for simultaneous elemental and structural imaging
- **8-BM-B**: Large area mapping, XANES imaging for chemical speciation
- **33-ID-C**: Post APS-U flagship coherent imaging, high-resolution ptychography

---

### Spectroscopy Beamlines

| Beamline | Name/Designation | Technique | Energy Range | Beam Size |
|----------|-----------------|-----------|-------------|-----------|
| **9-BM** | XSD-SPC | XANES, EXAFS | 2.1–23 keV | 0.5–2 mm |
| **20-BM** | XSD-SPC | XANES, EXAFS | 2.7–32 keV | 0.5–3 mm |
| **25-ID** | XSD-SPC | XANES, RIXS | 5–28 keV | 1–20 µm |

**Detectors**: Ion chambers (transmission), SDD/Vortex (fluorescence), Rowland spectrometer (RIXS)
**Data Output**: 1D energy-absorption spectra, µ-XANES imaging maps; 0.1–10 GB/dataset
**Research Focus**: Chemical speciation, oxidation states, local bonding environments

#### Key Capabilities
- **9-BM**: Tender/soft X-ray spectroscopy, light element edges (P, S, Cl, K, Ca)
- **20-BM**: Broad energy range for heavy metal speciation (Pb, As, Hg, U)
- **25-ID**: Microbeam spectroscopy for spatially-resolved speciation

---

### Scattering Beamlines

| Beamline | Name/Designation | Technique | Energy Range | Beam Size |
|----------|-----------------|-----------|-------------|-----------|
| **12-ID-B** | XSD-CMS | SAXS, WAXS | 7.9–14 keV | 20–200 µm |
| **12-ID-E** | XSD-DYN | XPCS | 7.4–12 keV | 1–20 µm |

**Detectors**: PILATUS 2M (SAXS), EIGER 500K (XPCS), Lambda 750K
**Data Output**: 2D scattering patterns, correlation functions; 10–100 GB/experiment
**Research Focus**: Nanostructure characterization, colloidal dynamics, biomolecular assembly

#### Key Capabilities
- **12-ID-B**: Simultaneous SAXS/WAXS, flow-through cells, automated sample changers
- **12-ID-E**: Post APS-U XPCS with dramatically enhanced coherent flux, dynamics from ns to hours

---

## Beamline-Modality Cross-Reference

| Modality | Beamlines |
|----------|-----------|
| Crystallography (MX/SSX) | 21-ID-D, 21-ID-F, 21-ID-G |
| Tomography (µCT/nano-CT) | 2-BM-A, 7-BM-B, 32-ID-B/C |
| XRF Microscopy | 2-ID-D, 2-ID-E, 8-BM-B |
| Ptychography | 2-ID-E, 33-ID-C |
| Spectroscopy (XAS) | 9-BM, 20-BM, 25-ID |
| Scattering (SAXS/XPCS) | 12-ID-B, 12-ID-E |

## Post APS-U Enhancements

All eBERlight beamlines benefit from APS-U improvements:

- **Crystallography**: Smaller, more intense beams enable microcrystallography and faster serial data collection
- **Imaging**: Higher coherence enables phase-contrast tomography as routine capability
- **Microscopy**: XRF sensitivity improves by 10-100×; ptychography becomes practical for biological samples
- **Spectroscopy**: Microbeam spectroscopy with nanoscale spatial resolution
- **Scattering**: XPCS measures faster dynamics due to dramatically higher coherent flux
