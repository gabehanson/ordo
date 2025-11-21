# Ordo

A structured and extensible pipeline for PLS and OPLS modeling in high-dimensional biological data.

Ordo was developed during the later stages of GH’s graduate work to provide a robust, generalizable, and transparent Python implementation of orthogonal partial least squares (OPLS). The package integrates preprocessing, feature selection, latent-variable modeling, variance partitioning, and model validation tools into a single coherent workflow.

## Core Features

OPLS and PLS modeling (via pyopls and scikit-learn)

OPLR for classification tasks

Bootstrap-based model validation with accuracy/Q² distribution estimates

Permutation testing for empirical significance

VIP score estimation with confidence intervals and loading-based sign

Scatter plots of latent variables (X-scores)

Optional mixOmics backend for exact variance explained calculations

Feature selection using LASSO with cross-validation

Class imbalance correction via balanced bootstrapping

Designed for high-dimensional biological data, including multiplexed imaging, transcriptomics, single-cell, and spatial datasets

## Requirements:
python >= 3.9

core dependencies
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
pyopls

## Installation
install with
pip install numpy pandas scipy matplotlib seaborn scikit-learn pyopls

## Optional: R / mixOmics backend for variance explained

The scikit-learn PLS implementation does not compute the true fraction of variance explained per component. Ordo includes an optional R bridge that uses the mixOmics pls() implementation (equivalent to the MATLAB algorithm) to compute:

variance explained in X

variance explained in Y

cumulative variance profiles
The dependencies for this are:
r-base
mixOmics
igraph
rpy2 >= 3.5

### Recommended installation using micromamba

to install, create a dedicated R enviornment using micromamba to provide a clean R installation with compatible shared libraries: 

micromamba create -n rmix r-base r-igraph r-mixomics -c conda-forge
micromamba activate rmix

Then point your python enviornment to this R installation by adding these environment variables to your python kernel (in kernel.json).
first find your kernel spec - from the terminal run:
jupyter kernelspec list
you will see a list of your jupyter environments with their respective path

find the one that corresponds to your environment and open the kernel.json file in that folder and add an env block like this (replace with your username):

{
  "argv": [
    "python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "my_env",
  "language": "python",
  "env": {
    "R_HOME=/Users/<username>/micromamba/envs/rmix/lib/R",
    "R_USER=/Users/<username>/micromamba/envs/rmix/lib/R",
    "R_SHARE_DIR=/Users/<username>/micromamba/envs/rmix/lib/R/share",
    "R_INCLUDE_DIR=/Users/<username>/micromamba/envs/rmix/lib/R/include",
    "R_LIBS=/Users/<username>/micromamba/envs/rmix/lib/R/library"
  }
}

now restart jupyter completely (not just the kernel)
and test this installation in python by running
    
import rpy2.robjects as ro
ro.r("library(mixOmics)")

if this loads cleanly, your R bridge is looking good
    
## Project Structure
    
ordo/
│
├── core.py              # PLS/OPLS modeling, preprocessing, VIPs, permutation tests
├── bootstrap.py         # Bootstrap PLS/OPLS with balanced resampling
├── mixomics.py          # Optional R/mixOmics functions for variance explained
├── plotting.py          # X-score scatter plots, VIP plots, CV distributions
├── utils.py             # Helper functions (e.g., subsampling)
│
├── __init__.py
├── README.md
└── examples/
      └── r2py_testing.ipynb

    
# Getting started:
    
from ordo import (
    preprocess_data,
    fit_pls_model,
    bootstrap_pls_analysis,
    variance_explained_mixomics,
    plot_scores_scatter
)

## pipeline 
### Preprocess
X_clean, y = preprocess_data(X, y)

### Fit PLS/OPLS
model, Z, opls = fit_pls_model(X_clean, y, n_components=2, orthogonalize=True)

### Bootstrap stability
results = bootstrap_pls_analysis(X_clean, y)

### Variance explained (optional R backend)
expl = variance_explained_mixomics(X_clean, y, ncomp=2)


    
Status

Actively developed and used in ongoing spatial immunology and transcriptomics research.
