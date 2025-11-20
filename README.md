📘 Human Activity Time-Series Clustering

Master’s Project Clustering of multivariate inertial sensor time series.

📄 Project Overview
This project investigates several unsupervised time-series clustering approaches to group human activities (walking, sitting, standing, stairs up, stairs down, lying) from tri-axial inertial signals.

We evaluate 2 main families of methods:
1. Shape-based clustering (time-domain)

DBA-KMeans (DTW Barycenter Averaging)

K-Shape (Shape-Based Distance — SBD)

2. Feature-based clustering

FFT + PCA + KMeans

FFT + UMAP + HDBSCAN

The goal is to compare their performance and understand which representations separate activities most effectively.

📂 Project Structure
├── data/
│   └── raw/                    # Original .txt sensor files
│
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   ├── dba_kmeans.ipynb        # DTW-based clustering (DBA K-Means)
│   ├── kshape.ipynb            # K-Shape clustering
│   ├── fft_kmeans.ipynb        # FFT + PCA + K-Means
│   └── hdbscan.ipynb           # FFT + UMAP + HDBSCAN
│
├── src/
│   ├── io.py                   # Data loading utilities
│   ├── preprocess.py           # Z-normalization, derivatives, windowing
│   ├── time_features.py        # Time-domain feature extraction
│   ├── fft.py                  # Fourier transform + spectral features
│   ├── reducers.py             # PCA / UMAP dimensionality reduction
│   ├── features.py             # Combines time + freq feature pipelines
│   ├── distances.py            # DTW / Soft-DTW / SBD distance matrices
│   ├── cluster_dba.py          # DBA-KMeans implementation
│   ├── cluster_kshape.py       # K-Shape implementation
│   ├── kmeans.py               # Classical K-Means wrapper
│   ├── hdbscan.py              # HDBSCAN wrapper
│   ├── eval.py                 # ARI, NMI, Purity, Silhouette (DTW)
│   ├── viz.py                  # t-SNE, UMAP, confusion matrices, plots
│   └── utils_io.py             # Saving runs, metrics, artifacts
│
├── results/
│   └── ...                     # Automatically saved experiment runs
│
├── report/
│   └── human_activity_clustering_report.tex
│
└── README.md


▶️ Quick Start
1. Install dependencies
pip install -r requirements.txt

2. Launch Jupyter notebooks
jupyter lab

3. Example — Run DBA-KMeans
from src.preprocess import z_norm_per_series
from src.cluster_dba import run_dba_kmeans

Xn = z_norm_per_series(X_train)
y_pred, model, centers = run_dba_kmeans(Xn, n_clusters=6)

📊 Key Findings

DTW and SBD methods perform well on highly dynamic activities.

Feature-based approaches (FFT + UMAP + HDBSCAN) achieve the best overall clustering quality.

Using a frequency representation + nonlinear manifold learning produces the clearest and most separable clusters.

👤 Authors

Project completed by Melissa MERABET and Ouarda BOUMANSOUR as part of a Master’s degree in Machine Learning & Data Science.