# SatMOC 2026 Session IV: Machine Learning Demo

This repository contains hands-on machine-learning examples prepared for **SatMOC 2026 Session IV**. The examples use a surface-ozone regression problem to demonstrate how common machine-learning models can be trained, evaluated, and interpreted for atmospheric-science applications.

The demo progresses from tree-based machine-learning models to a simple neural-network model:

1. **Random Forest**
2. **XGBoost**
3. **LightGBM**
4. **Multilayer Perceptron (MLP)**

The target variable in the example notebooks is surface ozone, `epa_o3`. The predictor variables include geolocation, topography, chemical reanalysis fields, radiation, meteorology, land-surface properties, population, nighttime lights, and temporal indicators.

---

## Repository structure

```text
satmoc-2026-session-IV/
├── README.md
├── DATA/
│   └── 2023_clean.csv                  # Input data file; not included if too large/private
├── satmoc2026_ML_Random_Forest.ipynb   # Random Forest ozone-regression demo
├── satmoc2026_ML_XGBoost.ipynb         # XGBoost ozone-regression demo
├── satmoc2026_ML_LightGBM.ipynb        # LightGBM ozone-regression demo
├── satmoc2026_ML_MLP.ipynb             # PyTorch MLP ozone-regression demo
├── datasets.py                         # Data reader, normalization, split, PyTorch Dataset
├── model.py                            # MLP model definition
└── train.py                            # MLP training, validation, testing, and metrics
```

---

## Learning objectives

After working through the notebooks, participants should be able to:

- prepare tabular atmospheric-composition data for machine-learning models;
- separate predictors, target variables, training data, validation data, and test data;
- train Random Forest, XGBoost, LightGBM, and MLP regression models;
- evaluate model performance using `R²`, `MAE`, `MSE`, `RMSE`, and mean bias;
- compare tree-based models and neural-network models conceptually and practically;
- inspect feature importance using built-in importance, permutation importance, or SHAP analysis.

---

## Data

The ozone dataset used in this session is available from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17502917.svg)](https://doi.org/10.5281/zenodo.17502917)

**Dataset:** High-resolution (1 km) daily surface MDA8 ozone dataset over the Continental USA, 2003–2024  
**Source:** https://zenodo.org/records/17502917  
**DOI:** https://doi.org/10.5281/zenodo.17502917

The full dataset is large and is organized by year as ZIP files. For the hands-on notebooks, download only the required year(s) or the prepared training table used in the session.

A recommended local directory structure is:

```text
satmoc-2026-session-IV/
├── data/
│   ├── raw/
│   │   ├── 2020.zip
│   │   └── 2021.zip
│   └── processed/
│       └── ozone_training_data.csv
├── satmoc2026_ML_Random_Forest.ipynb
├── satmoc2026_ML_XGBoost.ipynb
├── satmoc2026_ML_LightGBM.ipynb
├── satmoc2026_ML_MLP.ipynb
├── datasets.py
├── model.py
└── train.py

---

## Installation

Create a clean Python environment before running the notebooks. Python 3.10 or newer is recommended because current PyTorch releases require Python 3.10+.

```bash
conda create -n satmoc2026-ml python=3.10 -y
conda activate satmoc2026-ml
```

Install the core scientific Python packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn jupyter ipykernel
pip install xgboost lightgbm shap
```

### Install PyTorch

The MLP notebook uses PyTorch. Choose **one** of the following PyTorch installation options depending on your machine.

#### Option A: CPU-only installation

Use this option on laptops, desktops without an NVIDIA GPU, or teaching machines where GPU support is not needed.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option B: NVIDIA GPU installation with CUDA 12.6

Use this option on Linux/Windows machines with a compatible NVIDIA driver.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### Option C: NVIDIA GPU installation with CUDA 12.8

Use this option if your system requires a newer CUDA runtime supported by the official PyTorch binaries.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### Option D: macOS installation

For macOS, including Apple Silicon machines, use the default PyPI installation:

```bash
pip install torch torchvision torchaudio
```

Optional PyTorch-related packages:

```bash
pip install torchsummary
```

`torchsummary` is only used for printing a compact neural-network architecture summary. It is not required for model training.

After installation, verify that PyTorch works:

```bash
python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If `CUDA available` is `False`, the MLP notebook will still run on CPU, but training may be slower. For GPU use, make sure the NVIDIA driver is installed correctly. A full local CUDA toolkit is usually not required for running prebuilt PyTorch wheels.

Then register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name satmoc2026-ml --display-name "Python (satmoc2026-ml)"
```

Note: the notebooks use `root_mean_squared_error` from `scikit-learn`, so a recent `scikit-learn` version is recommended.

---

## Quick start

1. Clone the repository:

   ```bash
   git clone https://github.com/mzhou16-work/AMS-SatMOC-2026-session-IV
   cd satmoc-2026-session-IV
   ```

2. Place the ozone dataset in the expected location:

   ```bash
   mkdir -p DATA
   cp /path/to/2023_clean.csv DATA/2023_clean.csv
   ```

3. Start Jupyter:

   ```bash
   jupyter lab
   ```

4. Run the notebooks:

   ```text
   satmoc2026_ML_Random_Forest.ipynb
   satmoc2026_ML_XGBoost.ipynb
   satmoc2026_ML_LightGBM.ipynb
   satmoc2026_ML_MLP.ipynb
   ```

For portability, check the `data_dir` variable near the top of each notebook and make sure it points to:

```python
data_dir = './DATA/'
filename = '2023_clean.csv'
```

---

## Notebook descriptions

### 1. Random Forest demo

**Notebook:** `satmoc2026_ML_Random_Forest.ipynb`

This notebook demonstrates a classical ensemble-learning workflow using `RandomForestRegressor`. It includes:

- reading the ozone dataset;
- defining target and predictor variables;
- splitting the dataset into training and test subsets;
- imputing missing numerical and categorical values;
- one-hot encoding `modis_landtype`;
- training a Random Forest model;
- evaluating model skill with `R²`, `RMSE`, `MAE`, and mean bias;
- plotting observed versus predicted ozone;
- examining feature importance and permutation importance.

### 2. XGBoost demo

**Notebook:** `satmoc2026_ML_XGBoost.ipynb`

This notebook demonstrates gradient-boosted decision trees using `XGBRegressor`. It is useful for showing how boosting differs from bagging-based models such as Random Forest. It includes:

- the same ozone-regression input setup as the Random Forest notebook;
- preprocessing with `ColumnTransformer`, `SimpleImputer`, and `OneHotEncoder`;
- XGBoost model training with regularization and shrinkage;
- observed-versus-predicted plotting;
- model feature importance;
- permutation importance on a subset of the test data.

### 3. LightGBM demo

**Notebook:** `satmoc2026_ML_LightGBM.ipynb`

This notebook demonstrates LightGBM for efficient gradient boosting on tabular data. It includes:

- loading the ozone dataset;
- applying the same numerical/categorical preprocessing strategy;
- training an `LGBMRegressor`;
- evaluating model performance;
- plotting observed versus predicted ozone.

Before running this notebook, make sure the `data_dir` variable points to the local `DATA/` directory.

### 4. MLP demo

**Notebook:** `satmoc2026_ML_MLP.ipynb`

This notebook demonstrates a simple neural-network regression model using PyTorch. Compared with the tree-based notebooks, this workflow is more explicit about normalization, mini-batch training, validation, and training diagnostics.

The MLP workflow uses:

- `datasets.py` for data reading, quality control, normalization, one-hot encoding, data splitting, and PyTorch dataset construction;
- `model.py` for the `MLPRegressor` model;
- `train.py` for training, validation, testing, metric calculation, and inverse normalization of predictions.

The notebook includes:

- Gaussian normalization of continuous predictors and the target variable;
- one-hot encoding of categorical predictors;
- a train/validation/test split;
- a fully connected MLP with LeakyReLU activation;
- stochastic-gradient-descent training;
- learning-rate reduction on validation performance;
- training and validation curves for `R²`, `MAE`, `RMSE`, mean bias, and learning rate;
- optional SHAP analysis for neural-network feature attribution.

---

## Evaluation metrics

The notebooks use common regression metrics:

| Metric | Meaning |
|---|---|
| `R²` | Fraction of variance explained by the model |
| `MAE` | Mean absolute error |
| `MSE` | Mean squared error |
| `RMSE` | Root mean squared error; same unit as ozone |
| `MB` | Mean bias, calculated as prediction minus observation |

These metrics are useful together because a model can have good correlation but still have systematic bias, or low average error but poor performance during high-ozone events.

---

## Expected outputs

The notebooks generate diagnostic outputs such as:

- observed-versus-predicted scatter plots;
- printed regression metrics;
- feature-importance plots;
- permutation-importance plots;
- MLP training and validation curves;
- optional SHAP feature-importance files and plots.

The MLP notebook writes outputs under:

```text
O3_MLP_training/
├── MLP_SAVE_DATA/
├── MLP_CHECK_POINT/
└── MLP_SAVE_DATA/SHAP/
```

These generated outputs should usually not be committed to GitHub unless they are small and intentionally included as examples.

---

## Recommended `.gitignore`

A minimal `.gitignore` for this repository could include:

```text
__pycache__/
.ipynb_checkpoints/
*.pyc

DATA/
O3_MLP_training/
*.pkl
*.joblib
*.pt
*.pth
*.csv
```

If you want to include a small sample dataset, place it in a separate folder such as `sample_data/` and document it clearly.

---

## Notes for instructors

Suggested teaching sequence:

1. Start with Random Forest to introduce tree ensembles and feature importance.
2. Move to XGBoost and LightGBM to explain boosting, learning rate, regularization, and sequential error correction.
3. Use the MLP notebook to introduce representation learning, activation functions, mini-batch training, validation curves, and overfitting/underfitting.
4. Compare all models using the same target, predictors, and evaluation metrics.

A useful discussion point is that tree-based models are strong baselines for structured/tabular atmospheric datasets, while neural networks become especially powerful when extended to high-dimensional spatial, temporal, or multimodal data.

---

## Citation and acknowledgement

This material was prepared for **SatMOC 2026 Session IV** as a teaching example for machine learning in atmospheric-science applications.

When using or adapting this repository, please acknowledge the SatMOC 2026 training material and the original data sources used to construct `2023_clean.csv`.

---

## License

Add the appropriate license for your repository before public release. For teaching/demo repositories, common choices include MIT, BSD-3-Clause, or Apache-2.0. If the dataset has access restrictions, use a separate data-use statement and do not redistribute restricted data.
