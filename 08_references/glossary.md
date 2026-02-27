# Glossary of Synchrotron Science Terms

## A

**Absorption edge**: The energy at which X-ray absorption increases sharply due to the excitation of a core electron. Each element has characteristic edge energies (K, L, M edges).

**ALCF**: Argonne Leadership Computing Facility. DOE supercomputing center at Argonne hosting the Aurora exascale system.

**APS**: Advanced Photon Source. DOE synchrotron X-ray light source at Argonne National Laboratory.

**APS-U**: Advanced Photon Source Upgrade. $815M project replacing the storage ring with an MBA lattice, achieving up to 500× brighter X-ray beams.

## B

**Beam time**: Allocated time for an experiment at a synchrotron beamline. Typically 3-5 days per allocation.

**Beamline**: An experimental station at a synchrotron that receives X-rays from the storage ring and delivers them to the sample.

**BER**: Office of Biological and Environmental Research. DOE office funding eBERlight.

**Bluesky**: Python-based experiment orchestration framework for synchrotron beamlines.

**Bragg peak**: A sharp diffraction peak satisfying Bragg's law (nλ = 2d sin θ), indicating constructive interference from crystal planes.

## C

**CBF**: Crystallographic Binary File. Legacy file format for diffraction images.

**CDI**: Coherent Diffraction Imaging. Lensless imaging using coherent X-rays.

**Coherence**: The property of X-ray waves having a well-defined phase relationship. The APS-U dramatically increases coherent flux.

**CuPy**: GPU-accelerated NumPy library used by TomocuPy for GPU computation.

**CXI**: Coherent X-ray Imaging format. HDF5-based standard for ptychography and CDI data.

## D

**DBA**: Double-Bend Achromat. The original APS storage ring lattice design (3rd generation).

**DLSIA**: Deep Learning for Scientific Image Analysis. Framework developed at LBNL for synchrotron image analysis.

## E

**eBERlight**: enabling Biological and Environmental Research with light. DOE BER program at APS.

**Emittance**: A measure of the beam size and divergence of the electron beam in a synchrotron. Lower emittance = brighter X-rays. APS-U achieves 42 pm·rad.

**EPICS**: Experimental Physics and Industrial Control System. Software framework for controlling synchrotron equipment.

**EXAFS**: Extended X-ray Absorption Fine Structure. Oscillations in the X-ray absorption spectrum beyond ~50 eV above the edge, encoding bond distance and coordination information.

## F

**FBP**: Filtered Back Projection. Standard analytical algorithm for tomographic reconstruction.

**FICUS**: Facilities Integrating Collaborations for User Science. DOE program enabling multi-facility proposals.

**Flat field**: Image taken with X-ray beam on but no sample (used for normalization in tomography).

**Fresnel zone plate**: Diffractive optical element used to focus X-rays to nanometer-scale spots.

## G

**GMM**: Gaussian Mixture Model. Probabilistic clustering method used for XRF data analysis.

**Gridrec**: FFT-based implementation of FBP used as the default algorithm in TomoPy.

## H

**HDF5**: Hierarchical Data Format version 5. Self-describing file format widely used for synchrotron data. Supports hierarchical groups, large datasets, and compression.

## I

**INR**: Implicit Neural Representation. Neural network that maps coordinates to values, representing continuous fields.

**Ion chamber**: Gas-filled detector measuring X-ray beam intensity, used for normalization.

## K

**KB mirrors**: Kirkpatrick-Baez mirrors. Pair of orthogonal curved mirrors used to focus X-ray beams.

## M

**MAPS**: Microanalysis Toolkit. Primary XRF spectral analysis software at APS.

**MBA**: Multi-Bend Achromat. The APS-U storage ring lattice design (4th generation), using 7 bends per sector to minimize emittance.

**MX**: Macromolecular Crystallography. Determination of protein/nucleic acid 3D structures by X-ray diffraction.

**µCT**: Micro-Computed Tomography. X-ray CT imaging with micrometer-scale resolution.

## N

**NeXus**: HDF5-based data format standard for neutron, X-ray, and muon science. Defines standardized schemas.

**nnU-Net**: Self-configuring U-Net framework that automatically determines optimal segmentation architecture.

## O

**Ophyd**: Python library providing hardware abstraction for Bluesky-controlled devices.

**Otsu thresholding**: Automatic thresholding method that minimizes intra-class variance in a bimodal histogram.

## P

**PCA**: Principal Component Analysis. Linear dimensionality reduction technique used in ROI-Finder for feature extraction.

**Phase retrieval**: Computational recovery of the phase of an X-ray wave from measured intensity (which only records amplitude²).

**Ptychography**: Coherent imaging technique using overlapping scan positions to solve the phase problem and reconstruct both amplitude and phase images.

**PtychoNet**: CNN-based approach for fast ptychographic phase retrieval.

## R

**Radon transform**: Mathematical transform relating an object function to its line integrals (projections). Basis of CT reconstruction.

**ROI**: Region of Interest. Selected area for detailed measurement.

**ROI-Finder**: ML-guided tool for selecting regions of interest in XRF microscopy.

## S

**SAXS**: Small-Angle X-ray Scattering. Measures nanostructure (1-100 nm) from scattering at small angles.

**Sinogram**: Rearrangement of projection data where each row corresponds to one angle. The input format for reconstruction algorithms.

**SIREN**: Sinusoidal Representation Networks. Neural network using sine activations for representing continuous signals.

**SSX**: Serial Synchrotron Crystallography. Collecting diffraction data from many randomly oriented microcrystals.

## T

**TomocuPy**: GPU-accelerated tomographic reconstruction tool using CuPy. 20-30× faster than TomoPy.

**TomoGAN**: GAN-based denoising method for low-dose synchrotron tomography.

**TomoPy**: Standard Python package for tomographic data processing and reconstruction.

## U

**U-Net**: Encoder-decoder CNN architecture with skip connections, widely used for image segmentation.

**UMAP**: Uniform Manifold Approximation and Projection. Nonlinear dimensionality reduction for visualization.

## V

**Voxel**: 3D pixel — a volume element in a reconstructed tomographic volume.

## W

**WAXS**: Wide-Angle X-ray Scattering. Measures atomic/molecular-scale structure from scattering at wide angles.

## X

**XANES**: X-ray Absorption Near-Edge Structure. Fine structure in XAS spectra near the absorption edge, sensitive to oxidation state and coordination geometry.

**XAS**: X-ray Absorption Spectroscopy. Measures X-ray absorption as a function of energy, probing local atomic environment and chemical state.

**XPCS**: X-ray Photon Correlation Spectroscopy. Measures dynamics by analyzing temporal fluctuations of coherent X-ray speckle patterns.

**XRF**: X-ray Fluorescence. Characteristic X-rays emitted by elements when excited by incident X-rays, used for elemental mapping.

## Z

**Zone plate**: See Fresnel zone plate.
