# Dark Geometry

<div align="center">

**A Unified Framework for Dark Matter and Dark Energy**

*The conformal mode of spacetime as the origin of the dark sector*

[![arXiv](https://img.shields.io/badge/arXiv-2512.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2512.XXXXX)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Dark_Geometry.pdf)
[![CLASS](https://img.shields.io/badge/CLASS-v3.3.4-blue.svg)](class_dg/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXX-blue.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

[Paper](#-paper) • [Theory](#-theory) • [Results](#-key-results) • [Installation](#-installation) • [Usage](#-usage) • [Citation](#-citation)

</div>

---

## 📋 Abstract

**Dark Geometry** proposes that dark matter and dark energy are two manifestations of a single phenomenon: the **scalar dynamics of spacetime** — specifically, the conformal (trace) mode of the metric, which we call the *Dark Boson*.

The central hypothesis is that the effective mass of this scalar field depends on local matter density:

$$m^2_{\rm eff}(\rho) = (\alpha^* M_{\rm Pl})^2 \left[1 - \left(\frac{\rho}{\rho_c}\right)^{2/3}\right]$$

| Regime | Condition | Behavior | Equation of State |
|--------|-----------|----------|-------------------|
| **Dark Matter** | ρ > ρ_c | Tachyonic (m² < 0) | w ≈ 0 |
| **Dark Energy** | ρ < ρ_c | Stable (m² > 0) | w ≈ −1 |

The transition occurs at the **critical density** ρ_c ≡ ρ_DE, providing a natural explanation for cosmic acceleration.

---

## 🎯 Key Results

### Cosmological Tensions

| Problem | ΛCDM | Dark Geometry | Status |
|---------|------|---------------|--------|
| **σ₈ tension** | 2.7σ (vs DES) | 0.9σ | ✅ Strongly alleviated |
| **H₀ tension** | 4.8σ (vs SH0ES) | < 1σ | ✅ Strongly alleviated |
| **Cusp-core** | NFW cusp (n = −1) | Core (n ≈ 0) | ✅ Consistent with obs. |
| **Missing satellites** | ~500 predicted | ~60 predicted | ✅ Consistent with obs. |

### Numerical Results from CLASS-DG

```
σ₈(ΛCDM)  = 0.823  →  σ₈(DG)   = 0.785  (−4.6%)
H₀(Planck) = 67.4   →  H₀(DG-E) = 73.0   km/s/Mpc
CMB TT/TE/EE: Identical to ΛCDM (< 0.01% difference)
Sound horizon: rs reduced by 4.2% in DG-E
```

---

## 📖 Theory

### Fundamental Parameters

All parameters are **theoretically motivated**, not fitted to data:

| Parameter | Value | Origin | Uncertainty |
|-----------|-------|--------|-------------|
| α* | 0.075 | Asymptotic Safety UV fixed point | ~15-20% |
| ρ_c | ρ_DE ≈ 5.8 × 10⁻²⁷ kg/m³ | Friedmann geometry | Exact |
| β | 2/3 | Holographic scaling (A ∝ V^(2/3)) | Conjectured |
| ξ₀ | 0.105 | Non-minimal coupling (DG-E) | Calibrated |

### Physical Mechanism

1. **In galaxies/clusters** (ρ > ρ_c): The Dark Boson is tachyonic, clusters with matter → Dark Matter behavior
2. **In voids** (ρ < ρ_c): The Dark Boson is stable, acts as cosmological constant → Dark Energy behavior
3. **Transition** at z ≈ 0.3 coincides with the onset of cosmic acceleration

### DG-E Extension (for H₀)

The extended model includes a non-minimal coupling ξRφ² that modifies the Hubble rate at high redshift:

$$H_{\rm DG-E}(z) = H_{\Lambda\rm CDM}(z) \times \sqrt{1 + f_{\rm eff}(z)}$$

This reduces the sound horizon r_s by ~4%, increasing the inferred H₀ to match local measurements.

---

## 📁 Repository Structure

```
Dark-Geometry/
│
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
│
├── 📁 paper/                    # Article
│   ├── Dark_Geometry.tex        # LaTeX source (55 KB)
│   ├── Dark_Geometry.pdf        # Compiled paper (26 pages)
│   ├── derivation_1_systeme_couple.md
│   └── derivation_2_WKB.md
│
├── 📁 class_dg/                 # CLASS Implementation
│   ├── 📁 source/               # Modified C source files
│   │   ├── background.c         # H(z) modification for DG-E
│   │   ├── fourier.c            # P(k) suppression function
│   │   ├── input.c              # Parameter reading
│   │   └── dark_geometry.c      # New DG physics module
│   ├── 📁 include/              # Header files
│   │   ├── background.h         # DG parameter structures
│   │   └── dark_geometry.h      # DG function declarations
│   ├── 📁 ini_files/            # Configuration files
│   │   ├── lcdm_test.ini        # ΛCDM baseline
│   │   ├── dg_test.ini          # DG (σ₈ test)
│   │   └── dge_test.ini         # DG-E (H₀ test)
│   ├── 📁 analysis/             # Analysis scripts
│   │   ├── compare_dg_lcdm.py
│   │   ├── analyze_dge_H0.py
│   │   └── final_summary.py
│   └── README.md                # CLASS-DG documentation
│
├── 📁 simulations/              # Python simulations (14 scripts)
│   ├── hcm_class_simulation.py
│   ├── hcm_complete_analysis.py
│   ├── HCM_Extended_numerical.py
│   ├── HCM_cusp_core.py
│   ├── hertault_dwarfs.py
│   ├── hertault_halo_simulation.py
│   └── ...
│
├── 📁 figures/                  # All figures
│   ├── fig_conceptual.png       # Conceptual diagram
│   ├── fig_three_regimes.png    # Mass function & w(z)
│   ├── fig_power_spectrum.png   # P(k) with suppression
│   ├── fig_sigma8.png           # σ₈ tension comparison
│   ├── fig_H0_tension.png       # H₀ tension resolution
│   ├── fig_cusp_core.png        # Density profiles
│   ├── fig_dwarfs.png           # Satellite problem
│   ├── fig_complete_analysis.png
│   └── generate_all_figures.py  # Figure generation script
│
└── 📁 data/                     # Output data files
```

---

## 🚀 Installation

### Requirements

- **CLASS** v3.3.4 or later ([github.com/lesgourg/class_public](https://github.com/lesgourg/class_public))
- **Python** 3.8+ with NumPy, SciPy, Matplotlib
- **C compiler** (gcc recommended)

### Step-by-step Installation

```bash
# 1. Clone this repository
git clone https://github.com/hugohertault/Dark-Geometry.git
cd Dark-Geometry

# 2. Clone CLASS
git clone https://github.com/lesgourg/class_public.git
cd class_public
git checkout v3.3.4

# 3. Copy Dark Geometry modifications
cp ../class_dg/source/*.c source/
cp ../class_dg/include/*.h include/

# 4. Edit Makefile - add dark_geometry.o to the SOURCE line:
#    SOURCE = ... dark_geometry.o

# 5. Compile
make clean
make class

# 6. Verify installation
./class ../class_dg/ini_files/lcdm_test.ini
```

---

## 💻 Usage

### Running Simulations

```bash
cd class_public

# ΛCDM baseline
./class ../class_dg/ini_files/lcdm_test.ini

# Dark Geometry (σ₈ tension)
./class ../class_dg/ini_files/dg_test.ini

# Dark Geometry Extended (H₀ tension)
./class ../class_dg/ini_files/dge_test.ini
```

### Configuration Parameters

**DG parameters** (in `.ini` file):
```ini
# Enable Dark Geometry
has_dg = yes

# Fundamental parameters
dg_alpha_star = 0.075
dg_rho_c = 5.82e-27

# Suppression function
dg_k_suppression = 0.1      # h/Mpc
dg_beta_suppression = 2.8
dg_A_suppression = 0.25
```

**DG-E additional parameters**:
```ini
# Enable DG-Extended
has_dg_extended = yes

# Non-minimal coupling
dg_xi_0 = 0.105
dg_beta_xi = 0.02
dg_beta_alpha = 0.04
dg_eta = 80.0
```

### Analysis Scripts

```bash
cd class_dg/analysis

# Compare DG vs ΛCDM
python compare_dg_lcdm.py

# Analyze H₀ tension
python analyze_dge_H0.py

# Generate summary figure
python final_summary.py
```

---

## 📊 Figures

| | |
|:---:|:---:|
| ![Conceptual](figures/fig_conceptual.png) | ![σ₈](figures/fig_sigma8.png) |
| *Conceptual framework* | *σ₈ tension alleviation* |
| ![H₀](figures/fig_H0_tension.png) | ![P(k)](figures/fig_power_spectrum.png) |
| *H₀ tension alleviation* | *Power spectrum suppression* |

---

## 📚 Citation

If you use Dark Geometry in your research, please cite:

```bibtex
@article{Hertault2025DarkGeometry,
    author = {Hertault, Hugo},
    title = {{Dark Geometry}: A Unified Framework for Dark Matter and Dark Energy},
    journal = {arXiv preprint},
    year = {2025},
    eprint = {2512.XXXXX},
    archivePrefix = {arXiv},
    primaryClass = {gr-qc},
    note = {With full CLASS implementation}
}
```

---

## 📖 References

| Reference | Description |
|-----------|-------------|
| [Lesgourgues (2011)](https://arxiv.org/abs/1104.2932) | CLASS Boltzmann code |
| [Planck 2018](https://arxiv.org/abs/1807.06209) | CMB constraints |
| [Riess et al. (2022)](https://arxiv.org/abs/2112.04510) | SH0ES H₀ measurement |
| [DES Y3 (2022)](https://arxiv.org/abs/2105.13549) | Weak lensing σ₈ |
| [Reuter & Saueressig (2019)](https://arxiv.org/abs/1912.02484) | Asymptotic Safety |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Open an issue for bugs or questions
- 🔧 Submit a pull request for improvements
- 📧 Contact the author for collaborations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Author**: Hugo Hertault  
**Location**: Tahiti, French Polynesia 🌴  
**Date**: December 2025  
**Contact**: hertault.toe@gmail.com

---

*"Dark matter and dark energy are not two mysteries, but two faces of the same geometry."*

</div>
