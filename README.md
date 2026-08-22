Physics-Informed Conformal Prediction of Geopolymer Concrete Strength

This repository contains the data-processing, machine-learning, uncertainty-quantification, model-interpretation, and graphical user-interface resources developed for predicting the compressive strength of geopolymer concrete.

The framework combines raw mixture variables with physics-informed descriptors, applies physical constraints to the predictions, and supplements point estimates with conformal prediction intervals. A desktop graphical user interface (GUI) is included for both individual and batch predictions.

## Main Features

- Prediction of geopolymer concrete compressive strength
- Physics-informed feature engineering based on precursor and activator chemistry
- Nonnegative compressive-strength predictions
- Monotonic strength development with testing age
- Split conformal prediction
- Stabilized adaptive conformal prediction
- CV+ conformal prediction
- SHAP, accumulated local effects, permutation importance, and sensitivity analysis
- Applicability-domain assessment
- Model-supported input-range guidance
- Manual and batch prediction through a desktop GUI
- Point estimates with lower and upper 90% prediction bounds


## Installation

Python 3.12 is recommended because it is compatible with the tested package versions and the serialized deployment bundle.

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

Replace `USERNAME` and `REPOSITORY_NAME` with the final GitHub account and repository names.

### 2. Create a virtual environment

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tested core package versions are:

```text
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
xgboost==3.0.2
joblib==1.4.2
customtkinter==5.2.2
pillow==11.1.0
openpyxl==3.1.5
pyinstaller==6.12.0
```

On macOS, XGBoost may also require the OpenMP runtime:

```bash
brew install libomp
```

## Running the Modeling Notebook

Open `notebooks/Master_Geopolymer_Modeling_and_Analysis.ipynb` in JupyterLab, Jupyter Notebook, or Google Colab and execute the cells sequentially.

The notebook covers:

1. Database loading and preprocessing
2. Physics-informed feature engineering
3. Model training and evaluation
4. Physical-constraint implementation
5. Conformal prediction and interval evaluation
6. Model interpretation and sensitivity analysis
7. Applicability-domain assessment
8. Export of the final deployment bundle

If Google Colab is used, restart the runtime after installing or changing NumPy, scikit-learn, or XGBoost. Mixing binary packages from different installations can cause import or model-loading errors.

## Running the GUI

Place `geopolymer_final_deployment_bundle.joblib` beside the GUI script or select it using the **Select Model Bundle** button.

From the repository root, run:

```bash
python src/geopolymer_gui.py
```

The GUI provides:

- Manual prediction for an individual mixture
- Batch prediction from an Excel file
- Point estimates of compressive strength in MPa
- Lower and upper conformal prediction bounds
- Prediction-interval width
- Applicability-domain classification
- Supported and observed ranges for every input
- Warnings for missing, uncommon, or out-of-range inputs

## Input Validation

The GUI checks that all required variables are numeric and available. It also enforces the following precursor-composition requirement:

```text
SiO2 + Al2O3 + Fe2O3 + CaO <= 100 wt.%
```

Inputs are compared with the training database and classified as supported, uncommon, or outside the observed range. These ranges describe the model's data support; they should not be interpreted as universally optimal geopolymer mixture-design limits.

## Prediction Methods

Three conformal prediction approaches are available:

- **Split conformal prediction:** uses a held-out calibration set to construct prediction intervals.
- **Stabilized adaptive conformal prediction:** scales the interval according to the estimated local prediction difficulty.
- **CV+:** combines cross-validation predictions and residuals to construct prediction bounds.

The nominal coverage is 90%. This is a marginal coverage target and does not guarantee identical coverage for every precursor type, curing regime, or compositional subgroup.

## Batch Prediction

Use the GUI to download a blank Excel template containing the required column names. Enter one mixture per row without modifying the headers, upload the completed file, select a conformal method, and export the results.

The output includes the point estimate, lower and upper bounds, interval width, nominal coverage, applicability-domain information, and input-range assessment.

## Model and Data Availability

The deployment bundle contains the trained estimator and the preprocessing, physics-informed feature, conformal prediction, and applicability-domain objects required by the GUI.

Before publishing the repository, confirm that the compiled experimental database and trained model can be redistributed. If the complete database cannot be shared, provide a synthetic example file, variable definitions, preprocessing code, and instructions for authorized users to supply the source data independently.

## Important Use Notes

- Predictions are intended to support preliminary mixture screening and research.
- Predictions outside the applicability domain should be interpreted cautiously.
- Conformal intervals communicate predictive uncertainty but do not eliminate model risk.
- Model-supported ranges are not recommendations for optimal mixture design.
- Laboratory testing remains necessary before engineering implementation.
- Physicochemical interpretations derived from model behavior should not be treated as direct proof of reaction mechanisms.

## Reproducing the Manuscript Results

Run the master notebook from beginning to end using the package versions listed in `requirements.txt`. The notebook should generate the model-performance, conformal-prediction, interpretation, sensitivity, interaction, and applicability-domain results reported in the associated paper.

Random seeds and data partitions should remain unchanged when exact numerical reproduction is required.

## Citation

If you use this repository, please cite the associated paper:

```text
Behnood, A., [Coauthor names]. "[Final paper title]." [Journal], [Year].
DOI: [Insert DOI]
```

The citation will be updated after publication.

## License



## Contact

**Ali Behnood, Ph.D., MBA, P.E.**  
behnood.al@gmail.com

Questions, bug reports, and reproducibility issues may be submitted through the repository's GitHub Issues page.

