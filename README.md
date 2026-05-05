# Template-Free Geometric Analysis of Gravitational-Wave Events

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18228132.svg)](https://doi.org/10.5281/zenodo.18228132)

**Pipeline, dataset, and geometric classification catalog accompanying:**

> Burgos, M. E. *Template-Free Geometric Analysis of Gravitational-Wave Events* (2026).  
> Zenodo. [10.5281/zenodo.18228132](https://doi.org/10.5281/zenodo.18228132)

---

## Overview

This repository contains the complete implementation of the template-free geometric analysis method described in the paper, along with the resulting **geometric classification dataset** for 47 gravitational-wave events from the LIGO-Virgo O2 and O3 observing runs.

The method operates directly on whitened strain data from interferometric detector networks, without using waveform templates. It extracts coherent geometric modes via spectral operator decomposition and classifies events based on their intrinsic network-space coherence structure.

**This dataset fulfills the "Future Work" section of the paper** (Section 6), specifically:

> *"Systematic analysis of the full GWTC would establish a complete geometric taxonomy and reveal potential correlations with astrophysical source properties."*

---

## Key Visualizations

### Geometric Classification Plane (η vs Rotation)

<img src="img/eta_rotation_plane.png" width="700" alt="η vs Rotation plane">

Each point is a GW event. **η** measures geometric coherence (how clean the signal is in detector space). **Rotation** measures cross-phase stability (how much the geometry changes from inspiral to merger). The plane reveals four natural quadrants with distinct astrophysical properties.

### Event Distribution

<img src="img/class_distribution.png" width="700" alt="Class and source type distribution">

**94% of confident GWTC events are Class B⁺** (multi-component coherent geometry). Only 2 events achieve Class A (one-dimensional geometry). Zero false positives — all events show measurable geometric coherence.

### Geometric Method vs LIGO — Correlation Matrix

<img src="img/correlation_matrix.png" width="750" alt="Correlation matrix geometric vs LIGO">

The correlation matrix reveals that **geometric and LIGO parameters are largely orthogonal** (max cross-correlation r=0.52 between η and SNR — only 27% shared variance). The geometric method measures something fundamentally different from template-based parameter estimation.

### Geometric Coherence η — A Template-Free Signal Quality Metric

<img src="img/eta_coherence_floor.png" width="750" alt="η distribution and coherence floor">

**Left:** η distribution across 47 events. The coherence floor at η≈0.37 means all confident GWTC events show measurable geometric structure. **Right:** η vs LIGO SNR (r=+0.40) — geometric coherence correlates with but is not redundant with signal strength. Low-SNR events can still have clean geometry, and vice versa.

### Geometric Sub-Phases — Beyond LIGO's 3-Phase IMR

<img src="img/phases_comparison.png" width="750" alt="Geometric phases vs LIGO phases">

**Left:** Number of geometric phases detected per event. LIGO imposes exactly 3 phases by model construction. Our method **discovers** phases from data — from 1 to 7 depending on source complexity. **Right:** GW170814 comparison. The geometric method reveals 6 sub-phases within LIGO's standard 3-phase IMR framework.

### Template-Free Mass Estimation

<img src="img/mass_estimation.png" width="700" alt="Template-free mass estimation GW170814">

For GW170814, the ISCO frequency detected geometrically (η peak at 80 Hz) yields M≈55.0 M☉ — **within 1.4% of the GWTC value** — using zero waveform templates and ~2 minutes of computation. This demonstrates the potential for rapid, model-independent mass estimation.

### IMR Phase Portrait — GW170814

<img src="img/imr_phase_portrait.png" width="700" alt="IMR phase portrait for GW170814">

The geometric method detects the inspiral-merger-ringdown transition **without templates**. The ISCO appears as a peak in η at ~78 Hz (matching f_ISCO ≈ 4400/M). The merger appears as a sharp drop (Δη = -0.11). The ringdown shows quasinormal mode oscillations at the expected frequencies for a ~53 M☉ remnant.

<img src="img/imr_phase_portrait.png" width="700" alt="IMR phase portrait for GW170814">

The geometric method detects the inspiral-merger-ringdown transition **without templates**. The ISCO appears as a peak in η at ~78 Hz. The merger appears as a sharp drop (Δη = -0.11). The ringdown shows quasinormal mode oscillations. **Template-free mass estimate: M ≈ 55.0 M☉ (GWTC: 55.8 M☉, error -1.4%).**

## Dataset

**`data/geometric_classification_dataset.csv`** — 47 events, 51 columns including:

| Category | Columns | Description |
|---|---|---|
| **LIGO astrophysical** | 11 | Masses, SNR, spin, p_astro, source type (from GWTC) |
| **Geometric core** | 6 | η (coherence), λ₁/λ₂, λ₂/λ₃, geometric class (A/B⁺/B/C), geometry mode, symmetry |
| **Geometric extended** | 5 | Cross-phase rotation, detector trajectory, quadrant, η/SNR ratio |
| **Per-phase** | 6 | η per IMR phase, waveform/envelope correlation with LIGO strain |
| **Detector power** | 3 | H1/L1/V1 participation in merger phase |
| **IMR granular** | 12 | ISCO η peak, merger drop, QNM peak frequency, ringdown amplitude, Class A regions |
| **Geometric mass** | 5 | Template-free total mass estimate from ISCO detection (GW170814) |

### Key findings

- **Class distribution**: A: 2 (4%), B⁺: 44 (94%), B: 1 (2%), C: 0 (0%)
- **All 47 confident GWTC events show geometric coherence** (B⁺ or better — zero false positives)
- **GW170814**: Template-free mass estimate M ≈ 55.0 M☉ (GWTC: 55.8 M☉, error -1.4%)
- **GW191230_180458**: Second Class A event discovered (η=0.867)
- **Geometric phases**: 1-7 per event (mean 3.5), data-driven IMR sub-structure

## Pipeline

The analysis pipeline is in `pipeline/`:

| Script | Purpose |
|---|---|
| `narrowband.py` | Core geometric analysis per frequency band |
| `classify.py` | Geometric classification (A/B⁺/B/C) |
| `waveforms.py` | Geometric waveform reconstruction + LIGO comparison |
| `run_pipeline.py` | Orchestrator: 3-band narrowband → classify → waveform |
| `batch_all.py` | Batch processing for multiple events |
| `batch_imr.py` | Granular IMR phase analysis (30–500 Hz) |
| `projected_eigenvalues.py` | Static eigenvalue spectrum visualization |

### Quick start

```bash
pip install -r requirements.txt
cd pipeline
python run_pipeline.py --event GW170814
```

## Citation

If you use this software or dataset in your research, please cite:

```bibtex
@software{burgos2026gwgeometric,
  author    = {Burgos, Marcelo Ernesto},
  title     = {Template-Free Geometric Analysis of Gravitational-Wave Events},
  year      = {2026},
  doi       = {10.5281/zenodo.18228132},
  url       = {https://github.com/mburgc/gw-geometric-analysis}
}
```

## License

MIT License — see [LICENSE](LICENSE) file.

The accompanying dataset (`data/geometric_classification_dataset.csv`) is also distributed under MIT for maximum reusability.
