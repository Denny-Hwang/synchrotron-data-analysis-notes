---
doc_id: LAB-EM-001
title: Electron Microscopy — Sample Data
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/electron_microscopy/]
---

# Electron Microscopy (Cryo-EM, SEM, TEM) — Sample Data

## Status: External Only

Cryo-EM datasets are huge (single EMPIAR entries are commonly **100 GB – TB**). The free tier of this repository cannot host them. Use the curated list in `../../docs/external_data_sources.md` instead.

## Recommended starting points

| Source | What you get | Size | License |
|---|---|---|---|
| **EMPIAR-10025** (T20S proteasome, Topaz-Denoise paper) | 196 micrographs | ~50 GB | **CC0** (public domain) |
| **EMPIAR-10185** (apoferritin) | smaller test set | a few GB | CC0 |
| **EMPIAR-10028** (β-galactosidase) | classic benchmark | ~10 GB | CC0 |

Each is hosted at https://www.ebi.ac.uk/empiar/ and downloadable via:

```bash
# Globus, Aspera, or direct HTTPS — see EMPIAR docs per entry
wget -r -np -nH --cut-dirs=4 \
    "ftp://ftp.ebi.ac.uk/empiar/world_availability/10025/data/raw/"
```

## EMPIAR licence notes

EMPIAR data is **CC0** (public domain dedication). You may redistribute, modify, and reuse without attribution requirement — though scientific citation is expected.

> Iudin, A., Korir, P. K., Salavert-Torres, J., Kleywegt, G. J., & Patwardhan, A. (2016).
> EMPIAR: a public archive for raw electron microscopy image data.
> *Nature Methods, 13*(5), 387–388.
> https://doi.org/10.1038/nmeth.3806

## Note for Interactive Lab developers

For a Topaz-Denoise demo within the Streamlit app, fetch a single EMPIAR-10025 movie slice on first run via `pooch` (with hash verification), cache it locally, and run inference with the pretrained `unet-3d` Topaz weights. Do **not** redistribute the raw movie within this repository.
