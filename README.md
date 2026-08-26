# Geopolymer Concrete Strength Predictor — Binder-Revised GUI

This desktop application uses the final binder-revised machine-learning workflow to report:

- a nonnegative compressive-strength point estimate;
- lower and upper conformal prediction bounds;
- prediction-interval width and nominal coverage;
- applicability-domain status and distance;
- supported, uncommon, and outside-observed input-range diagnostics; and
- manual and batch Excel/CSV prediction.

## Required model file

Run `Geopolymer_Modeling_Analysis.ipynb` completely. Copy the generated file below from `master_results/models/` into this GUI folder:

`geopolymer_final_deployment_bundle.joblib`

If the notebook generates `geopolymer_final_deployment_bundle_38_features.joblib`, copy it into this folder and rename the copied file to `geopolymer_final_deployment_bundle.joblib`. The file must be the newly exported binder-revised bundle; renaming an older bundle does not update its contents.

## macOS setup

1. Keep every supplied file in this folder.
2. Right-click `setup_macos.command`, select **Open**, and approve it if macOS asks.
3. When setup finishes, double-click `run_gui.command`.
4. If the model is not loaded automatically, click **Select Model Bundle** and select the 38-feature `.joblib` file.

The setup uses Homebrew Python 3.12, installs Tk support and `libomp`, creates a local `.venv`, and installs package versions compatible with the latest binder-revised bundle, including pandas 3.0, scikit-learn 1.8, and XGBoost 3.1. Python 3.13/3.14 is intentionally avoided because binary packages and saved model objects are more likely to be incompatible.

## Build a standalone macOS application

After the GUI runs successfully from Python, double-click `build_macos_app.command`. The application will be created at:

`dist/Geopolymer_Strength_Predictor_38_Features.app`

The application is built for the Mac architecture used for compilation. Build on Apple Silicon for Apple Silicon Macs and on Intel for Intel Macs. macOS may require right-clicking the application and selecting **Open** on first launch.

## Inputs

The GUI requests the 24 retained raw predictors used by the final model. It calculates the 14 physics-informed features internally. Total binder is not entered separately; it is calculated from OPC, fly ash, GGBFS, silica fume, metakaolin, and other SCM contents.

The GUI blocks physically invalid entries, including negative inputs, nonpositive age or total binder, a major-oxide sum above 100 wt.%, inconsistent sodium-silicate fractions, and dry NaOH exceeding the NaOH-solution mass.

## Batch prediction

Open the **Batch Prediction** tab and select **Download Input Template**. Keep the exact required column names. Each spreadsheet row represents one mixture and testing age. Results can be exported to Excel or CSV.

## Troubleshooting

- **Model load error:** recreate the bundle using the same environment used to train the model, then rebuild the app.
- **`libomp.dylib` error:** run `brew install libomp`.
- **`_tkinter` error:** run `brew install python-tk@3.12` and rerun setup.
- **macOS blocks the scripts:** right-click the `.command` file, choose **Open**, then confirm.
- **Bundle not found:** place the exact 38-feature bundle beside `geopolymer_gui.py`, or select it from the GUI.
