# Raman Surface Map Analysis Framework

A Python framework for automated analysis of Raman surface mapping data in condensed matter systems. Developed as a Senior Honours Project at the University of Edinburgh (School of Physics and Astronomy).

---

## Overview

Raman surface mapping produces large volumes of spectral data — hundreds to thousands of individual spectra — that are difficult to interpret manually. This framework provides four analytical tools to address this:

| Module | Description |
|--------|-------------|
| **Spatial Visualiser** | Interactive heatmap for exploring spectral intensity distributions across a sample surface |
| **Distinct Spectra Estimation** | PCA and noise-whitened HFC methods to estimate the number of spectrally distinct sources |
| **Spectral Extraction** | Non-negative matrix factorisation (NMF) to extract and spatially map distinct spectral components |
| **Compound Prediction** | 1D CNN (reconstructed from Liu et al., 2017) for Raman spectrum classification; achieves **90.7% test accuracy** on the RRUFF mineral database |

---

## Visualisations

### Interactive Spatial Analysis
The interactive heatmap provides a method for exploring a spectral cube.

![Spatial Visualiser UI](outputs/heatmap_sb.png)
*Figure 1: The snapshot of the interactive heatmap showing the spectral intenstiy distribution across the sample surface.*

### Spectral Extraction
NMF is used to extract spectraly distinct sources from the sample.

![NMF Decomposition](outputs/sb2.1.png)
*Figure 2: The snaphot of the extractred distinct spectra from the entire sample.*

![NMF Decomposition](outputs/sb2.2.png)
*Figure 3: The snapshot of the heatmap view of the extracted spectra.*

---

## Project Structure

```
.
├── src/
│   ├── analysis/          # Phase number estimation (PCA, NWHFC)
│   ├── cnn/               # CNN model, training, evaluation, and prediction
│   ├── data/              # Data loading and grid utilities
│   └── visualisation/     # Heatmap, phase decomposition, and prediction viewers
├── artifacts/
│   ├── models/            # Saved Keras model files
│   ├── weights/           # Saved model weights
│   ├── encoders/          # Label encoders for compound classes
│   └── metadata/          # Wavenumber range files
├── notebooks/             # Exploratory scripts and examples
├── scripts/               # Entry-point plotting scripts
├── outputs/               # Generated figures and plots
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/raj2004n/Raman-Deep-Learning.git
cd Raman-Deep-Learning
pip install -r requirements.txt
```

Python 3.12 is recommended.

---

## Usage

### Spatial Visualiser

Launches an interactive heatmap for exploring a Raman surface map. The rolling spectral window, intensity slider, and logarithmic scale toggle allow real-time exploration of spatial and spectral features.

```python
from src.visualisation.view_heatmap import show_heatmap_viewer
import ramanspy as rp

hsi_cube = rp.SpectralImage(data, spectral_axis)
show_heatmap_viewer(hsi_cube)
```

### Distinct Spectra Estimation

Estimates the number of spectrally distinct sources using PCA (scree and 80% variance criteria) and the noise-whitened Harsanyi-Farrand-Chang method across a range of false alarm rates.

```python
from src.analysis.phase_number_est import estimate_phase_number

n_components, confidence = estimate_phase_number(hsi_cube)
print(f"Estimated {n_components} distinct spectra ({confidence} confidence)")
```

### Spectral Extraction (NMF)

Extracts distinct spectral components and displays their spatial abundance maps. The number of components can be set manually or estimated automatically.

```python
from src.visualisation.view_phase_decomp import show_spectra_ext_viewer

# n_components=-1 triggers automatic estimation
show_spectra_ext_viewer(hsi_cube, n_components=-1, start=200, end=1200)
```

### Compound Prediction (CNN)

Predicts the mineral compound for a given spectrum using the trained 1D CNN. The model was trained on the RRUFF `poor_unoriented` dataset (1639 spectra, 533 classes).

```python
from src.visualisation.view_predict import show_predict_viewer

show_predict_viewer(hsi_cube)
```

---

## Methods

### Preprocessing Pipeline

Applied prior to estimation and extraction:

- **Whitaker-Hayes despiking** — removes cosmic ray artefacts
- **Asymmetric least squares (ASLS) baseline correction** — removes fluorescence background
- **Min-max normalisation** — scales spectra to [0, 1] for NMF compatibility

### Distinct Spectra Estimation

Two methods are implemented and compared:

- **PCA (scree method)** — identifies the elbow in the eigenvalue spectrum; found to be the most reliable estimator for condensed matter Raman data
- **PCA (80% explained variance)** — tends to overestimate; sensitive to SNR and spectral range
- **Noise-whitened HFC (NWHFC)** — hyperspectral virtual dimensionality method; evaluated across false alarm rates from 10⁻² to 10⁻⁷. Found to be volatile and unreliable without sample-specific tuning for this data type

### NMF Extraction

- **Initialisation:** NNDSVDA (non-negative double SVD, average variant) for dense data
- **Solver:** Coordinate descent
- **Loss:** Frobenius norm (least-squares); note that Raman noise is Poisson-distributed, so KL-divergence loss may be more appropriate for future work
- **Max iterations:** 10,000

### CNN Architecture

Reconstructed from Liu et al., *Analyst*, 2017. A LeNet-variant 1D CNN:

- Three convolutional blocks (16 → 32 → 64 kernels, sizes 21 → 11 → 5), each followed by batch normalisation, Leaky ReLU, and max-pooling (stride 2)
- Fully connected layer (2048 neurons) with batch normalisation and dropout (0.5)
- Output layer with softmax over *C* classes
- Trained with class-weighted categorical cross-entropy and Adam optimiser
- Data augmentation: spectral shift, noise injection, and linear combination of same-class spectra (4.4× training set increase)

**Result:** 90.7% test accuracy on 493 test spectra across 533 classes (cf. 93.3% reported by Liu et al.)

---

## Results

| Method | Result |
|--------|--------|
| Spatial visualiser | Successfully captures surface heterogeneity; rolling window enables region-specific analysis |
| PCA scree | Correctly predicted 2 distinct spectra for both test samples in the fingerprint region |
| NMF extraction | Successfully separated diamond anvil cell background from sample signal |
| CNN (poor unoriented) | **90.7% test accuracy**, 200 epochs, trained on NVIDIA RTX 3060 |
| CNN (excellent oriented) | **98.1% test accuracy** |

---

## Dependencies

Key libraries used:

- [RamanSPy](https://github.com/bagheria-lab/ramanspy) — Raman data structures, NMF, preprocessing pipelines
- [scikit-learn](https://scikit-learn.org/) — PCA, StandardScaler
- [PySptools](https://pysptools.sourceforge.io/) — NWHFC virtual dimensionality
- [Keras](https://keras.io/) — CNN implementation
- [NumPy](https://numpy.org/) — numerical operations

See `requirements.txt` for the full list.

---

## References

- Liu et al., "Deep convolutional neural networks for Raman spectrum recognition: a unified solution", *Analyst*, 2017
- Chang & Du, "Estimation of number of spectrally distinct signal sources in hyperspectral imagery", *IEEE TGRS*, 2004
- Lee & Seung, "Algorithms for non-negative matrix factorization", *NeurIPS*, 2000
- Georgiev et al., "RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis", *Analytical Chemistry*, 2024

---

## Author

**Raj Negi** — MPhys Computational Physics, University of Edinburgh
[github.com/raj2004n](https://github.com/raj2004n)