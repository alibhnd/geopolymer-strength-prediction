[README.md](https://github.com/user-attachments/files/31445796/README.md)
# Physics-Informed Machine Learning for Geopolymer Concrete Strength

This repository contains the reproducible modeling workflow and practical deployment resources developed for the paper:

> **A trustworthy physics-informed machine-learning framework for geopolymer compressive-strength prediction**

The framework combines machine-learning regression, physics-informed feature engineering, an age-monotonicity constraint, conformal prediction, model interpretation, sensitivity analysis, applicability-domain assessment, and a desktop graphical user interface (GUI).

## Main capabilities

- Predicts geopolymer concrete compressive strength.
- Uses 24 retained raw predictors describing precursor chemistry, binder constituents, aggregates, activator chemistry, water, curing conditions, and testing age.
- Generates 14 physics-informed descriptors, producing a 38-feature hybrid representation.
- Compares Random Forest, support vector regression, artificial neural networks, and XGBoost.
- Compares Optuna, Grey Wolf Optimizer (GWO), and Galactic Field Optimization (GFO).
- Enforces nonnegative compressive-strength predictions and a nondecreasing response with testing age.
- Generates point predictions and conformal lower and upper prediction bounds.
- Provides SHAP, permutation-importance, accumulated-local-effects, sensitivity, interaction, robustness, and applicability-domain analyses.
- Includes a standalone desktop GUI for manual and batch prediction.

## Repository structure

```text
PIML-Geopolymer-Strength/
├── README.md
├── LICENSE
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── Geopolymer_Modeling_Analysis.ipynb
│   └── Geopolymer_Descriptive_Statistics.ipynb
│
├── data/
│   ├── geopolymer.xlsx
│   └── README.md
│
├── gui/
│   ├── geopolymer_gui.py
│   ├── README_GUI.md
│   └── assets/
│
├── models/
│   └── README.md
│
├── results/
│   ├── figures/
│   └── tables/
│
└── supplementary/
    └── Supplementary_Information_PIML_Geopolymer.docx
```

The exact contents can be adjusted, but source code, data, models, results, and supplementary files should be stored in folders on the **main branch**, not in separate branches.

## Recommended environment

- Python 3.12
- macOS, Linux, or Windows
- At least 16 GB RAM recommended for the full workflow
- A local Jupyter environment is recommended for the final analysis because the complete optimization and interpretation workflow can run for an extended period.

## Installation using Conda

```bash
conda env create -f environment.yml
conda activate piml-geopolymer
python -m ipykernel install --user \
  --name piml-geopolymer \
  --display-name "Python 3.12 (piml-geopolymer)"
jupyter lab
```

On macOS, if XGBoost reports that `libomp.dylib` cannot be loaded, install the OpenMP runtime:

```bash
brew install libomp
```

## Installation using `venv` and pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name piml-geopolymer \
  --display-name "Python 3.12 (piml-geopolymer)"
jupyter lab
```

On Windows, activate the environment using:

```powershell
.venv\Scripts\activate
```

## Dataset

Place `geopolymer.xlsx` in the project root or in the `data/` directory and update `FILE_PATH` in the notebook accordingly. The final modeling workflow expects worksheet `Sheet 2` unless this configuration is changed.

The database contains multiple strength records obtained at different testing ages and curing conditions. The workflow retains 24 raw predictors and excludes five redundant source columns from direct model input. Total binder is reconstructed internally from OPC, fly ash, GGBFS, silica fume, metakaolin, and other supplementary cementitious materials; it is not supplied as an independent raw predictor.

Before distributing the dataset, confirm that redistribution is permitted by the license and terms of the original database. If redistribution is restricted, replace the workbook with a `data/README.md` containing the source citation, access link, variable definitions, and preprocessing instructions.

## Running the modeling notebook

1. Activate the project environment.
2. Open `notebooks/Geopolymer_Modeling_Analysis.ipynb`.
3. Select the `Python 3.12 (piml-geopolymer)` kernel.
4. Set the dataset path in the centralized configuration cell.
5. Use `FAST_MODE = True` for an initial diagnostic run.
6. Restart the kernel after the diagnostic run.
7. Set `FAST_MODE = False` and use **Run All** for the final manuscript analysis.

Downstream notebook cells depend on objects created earlier; individual cells should not be run out of order. The complete workflow writes figures, tables, and the deployment model bundle to the configured results directory.

On macOS, this command can prevent sleep during a long run:

```bash
caffeinate -i
```

## Reproducibility settings

The final analysis uses a fixed random seed and a held-out test set. The notebook records the following major settings in a centralized configuration cell:

- Random seed: 42
- Held-out test fraction: 20%
- Cross-validation folds: 5
- Benchmark Optuna trials: 75 in the full run
- Optimizer-comparison budget: 150 evaluations per optimizer in the full run
- Nominal conformal coverage: 90%
- Conformal bootstrap repetitions: 1,000 in the full run

Software versions, operating system, processor architecture, and stochastic algorithms can cause small numerical differences. Manuscript tables should be generated from one complete `FAST_MODE = False` execution.

## Running the GUI

The GUI requires the deployment bundle produced by the final modeling notebook. Use the filename:

```text
geopolymer_final_deployment_bundle.joblib
```

Place the bundle in the `models/` folder or beside `geopolymer_gui.py`, depending on the path defined in the GUI source code. Then run:

```bash
python gui/geopolymer_gui.py
```

The interface reports:

- Point estimate of compressive strength
- Lower and upper conformal prediction bounds
- Prediction-interval width
- Model-supported and observed input ranges
- Applicability-domain status
- Major-oxide summation check
- Manual and batch prediction options

The GUI does not replace engineering judgment. Predictions for out-of-domain mixtures or inputs outside the observed ranges should be treated cautiously.

## Large deployment bundle

The `.joblib` bundle may exceed GitHub's browser-upload limit. Recommended options are:

1. Attach the bundle to a GitHub Release.
2. Archive it on Zenodo and provide the DOI in `models/README.md`.
3. Use Git Large File Storage if the repository is configured for Git LFS.
4. Regenerate it by running the final notebook.

The supplied `.gitignore` excludes deployment bundles by default to prevent accidental commits. If Git LFS is configured, remove the corresponding ignore rule and track the bundle through `.gitattributes`.

## Results and supplementary information

The `results/` directory should contain only the final tables and publication-quality figures supporting the manuscript. Intermediate outputs, debugging files, cached objects, and obsolete result versions should not be committed.

The `supplementary/` directory contains analyses not reproduced as tables or figures in the main manuscript, including additional data diagnostics, robustness checks, uncertainty audits, applicability-domain results, stratified errors, and interaction grids.

## Citation

If you use the code, model, data-processing workflow, or GUI, please cite the associated paper. Complete the repository's `CITATION.cff` after the article receives its final bibliographic information and DOI.

## License

Code use is governed by the repository's `LICENSE` file. The dataset and cited third-party resources may be subject to separate terms and are not automatically covered by the software license.

## Contact

**Ali Behnood, Ph.D., MBA, P.E.**  
Email: behnood.al@gmail.com

