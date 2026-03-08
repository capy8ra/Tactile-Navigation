# Tac-Nav: Tactile-Based Human Intent Recognition for Robot Assistive Navigation

Official code for the ICRA 2026 paper:

> **Tactile-Based Human Intent Recognition for Robot Assistive Navigation**
> Shaoting Peng, Dakarai Crowder, Wenzhen Yuan, Katherine Driggs-Campbell
> University of Illinois Urbana-Champaign
> [[arXiv]](https://arxiv.org/abs/2509.16353)

---

## Overview

Tac-Nav uses a cylindrical tactile skin mounted on a Stretch 3 robot to recognize five navigation intents — *turn left*, *turn right*, *speed up*, *stop*, *neutral* — purely from touch.

The key algorithmic contribution is the **Cylindrical Kernel SVM (CK-SVM)**, which replaces the standard RBF distance with a *Cylindrical Distance* that is robust to rotational shifts in the user's grasp:

```
d_C(x1, x2) = min_{s=0..k-1} ( ||x1 - shift_s(x2)||^2_F + exp(min(s, k-s) / delta) - 1 )
```

---

## Repository Structure

```
Tactile-Navigation/
├── src/
│   ├── ck_svm.py        # CK-SVM algorithm (main contribution)
│   ├── classifiers.py   # Baseline classifiers: RBF-SVM, MLP, MDCM, CNN
│   └── features.py      # Shared feature extraction (mean, max, std, spatial gradient)
├── scripts/
│   ├── generate_data.py # Generate simulated dataset
│   ├── evaluate.py      # Train & evaluate all classifiers
│   └── visualize_data.py# Visualize a single trial as heatmaps
├── data/
│   ├── simulated/       # Pre-generated .npy files (200 samples, 5 classes)
│   └── final_data/      # Real-world collected data
└── results/             # Accuracy comparison plot (auto-created on run)
```

---

## Installation

```bash
git clone https://github.com/<your-username>/Tactile-Navigation.git
cd Tactile-Navigation
pip install -r requirements.txt
```

---

## Quick Start

**1. Evaluate on the pre-generated simulated data**

```bash
python scripts/evaluate.py
```

Trains CK-SVM and all four baselines on `data/final_data/` and prints a results table. Saves an accuracy comparison bar chart to `results/`.

**2. Evaluate on real-world data**

```bash
python scripts/evaluate.py --data_dir data/final_data
```

**3. Multiple random splits**

```bash
python scripts/evaluate.py --n_runs 10
```

Each run uses an independent random seed for the train/test split.

**4. Re-generate the simulated dataset**

```bash
python scripts/generate_data.py --output_dir data/final_data --n_trials 40
```

**5. Visualize a trial**

```bash
# Snapshot grid
python scripts/visualize_data.py --action forward --trial 0

# Snapshot grid + animation
python scripts/visualize_data.py --action left --trial 3 --animate
```

Valid actions: `forward`, `backward`, `left`, `right`, `static`

---

## Feature Extraction

Each raw sample `(T, 11, 5)` is collapsed into **4 spatial feature maps** `(11, 5, 4)` that preserve the sensor's cylindrical topology:

| # | Feature | Description |
|---|---|---|
| 0 | Mean activation | Average pressure distribution over time |
| 1 | Max activation | Peak pressure over time |
| 2 | Std activation | Temporal variability of pressure |
| 3 | Spatial gradient | Edge magnitude of the mean pressure map |

Flattened to a 220-dim vector for SVM and MLP; kept as a spatial tensor for CK-SVM.

---

## Classifiers

| Classifier | Description |
|---|---|
| **CK-SVM** | Proposed method — SVM with cylindrical kernel robust to rotational grasp shifts |
| RBF-SVM | Standard SVM with RBF kernel on flattened features |
| MLP | 3-layer MLP on standardized features |
| MDCM | Minimum Distance to Covariance Mean on the Riemannian manifold |
| CNN | Convolutional network on flattened features |

---

## Sensor & System

- **Sensor**: machine-knitted resistive tactile skin, 11 × 5 sensing grid, ~50 Hz
- **Handle**: 3-D printed cylindrical holder, ⌀ 7.4 cm × 15 cm
- **Robot**: Hello Robot Stretch 3, controlled via ROS 2
- **Data format**: each sample is `(T, 11, 5)` — T time steps × 11 rows × 5 cols

---

## Citation

```bibtex
@article{peng2025tacnav,
  title   = {Tactile-Based Human Intent Recognition for Robot Assistive Navigation},
  author  = {Peng, Shaoting and Crowder, Dakarai and Yuan, Wenzhen and Driggs-Campbell, Katherine},
  journal = {arXiv preprint arXiv:2509.16353},
  year    = {2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
