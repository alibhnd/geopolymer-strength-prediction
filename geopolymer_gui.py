"""Geopolymer Concrete Strength Predictor desktop application.

The application expects ``geopolymer_final_deployment_bundle.joblib`` beside
this script (or bundled into the executable). It supports manual and batch
prediction and reports physically constrained conformal prediction intervals.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


APP_TITLE = "Geopolymer Concrete Strength Predictor"
BUNDLE_NAME = "geopolymer_final_deployment_bundle.joblib"
BACKGROUND_NAME = "background.jpg"
OXIDE_FEATURES = ["SiO2", "Al2O3", "Fe2O3", "CaO"]
OXIDE_SUM_MAX = 100.0

FRIENDLY_LABELS = {
    "SiO2": "SiO₂ content (wt.%)",
    "Al2O3": "Al₂O₃ content (wt.%)",
    "Fe2O3": "Fe₂O₃ content (wt.%)",
    "CaO": "CaO content (wt.%)",
    "Coarse aggregate (kg/m3)": "Coarse aggregate (kg/m³)",
    "Fine aggregate (kg in 1m3 mix)": "Fine aggregate (kg/m³)",
    "OPC (kg/m3)": "Ordinary Portland cement (kg/m³)",
    "FA (kg/m3)": "Fly ash (kg/m³)",
    "GGBFS (kg/m3)": "GGBFS (kg/m³)",
    "SF (kg/m3)": "Silica fume (kg/m³)",
    "MK (kg/m3)": "Metakaolin (kg/m³)",
    "Other SCM (kg/m3)": "Other SCMs (kg/m³)",
    "Total Na2SiO3 (kg in 1m3 of mix)": "Sodium silicate solution (kg/m³)",
    "Na2O (l)": "Na₂O fraction in liquid activator",
    "SiO2 (l)": "SiO₂ fraction in liquid activator",
    "Total NaOH (kg in 1m3 mix)": "Sodium hydroxide solution (kg/m³)",
    "Concentration (M) NaOH": "NaOH concentration (M)",
    "NaOH (Dry)": "Dry NaOH content (kg/m³)",
    "Additional water (kg in 1m3 mix)": "Additional water (kg/m³)",
    "Superplasticizer (kg in 1m3 mix)": "Superplasticizer (kg/m³)",
    "Total water (in solutions + additional) (kg in 1m3 mix)": "Total water (kg/m³)",
    "Initial curing time (day)": "Initial curing duration (days)",
    "Initial curing temp (C)": "Initial curing temperature (°C)",
    "Age_days": "Testing age (days)",
}


def feature_group(feature: str) -> str:
    if feature in {"SiO2", "Al2O3", "Fe2O3", "CaO"}:
        return "Precursor composition"
    if "aggregate" in feature.lower():
        return "Aggregates"
    if feature in {
        "OPC (kg/m3)", "FA (kg/m3)", "GGBFS (kg/m3)",
        "SF (kg/m3)", "MK (kg/m3)", "Other SCM (kg/m3)",
    }:
        return "Binder constituents"
    if feature in {
        "Additional water (kg in 1m3 mix)",
        "Superplasticizer (kg in 1m3 mix)",
        "Total water (in solutions + additional) (kg in 1m3 mix)",
    }:
        return "Water and admixture"
    if feature in {"Initial curing time (day)", "Initial curing temp (C)", "Age_days"}:
        return "Curing and testing"
    return "Activator chemistry"


def compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.4g}"


def resource_path(name: str) -> Path:
    """Return a resource path that works in Python and PyInstaller."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / name


def physical_predict(model, frame: pd.DataFrame, lower_bound: float = 0.0) -> np.ndarray:
    prediction = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    return np.maximum(prediction, lower_bound)


def empirical_quantile(values, q, axis=None, method="linear"):
    try:
        return np.quantile(values, q, axis=axis, method=method)
    except TypeError:
        return np.quantile(values, q, axis=axis, interpolation=method)


def cvplus_quantile_levels(n: int, alpha: float) -> tuple[float, float]:
    lower = max(0.0, math.floor(alpha * (n + 1)) / n)
    upper = min(1.0, math.ceil((1.0 - alpha) * (n + 1)) / n)
    return lower, upper


class PredictionBackend:
    """Loads the saved bundle and reproduces the notebook prediction logic."""

    def __init__(self, bundle_path: str | Path):
        self.bundle_path = Path(bundle_path)
        self.bundle = joblib.load(self.bundle_path)
        self.model = self.bundle["final_model"]
        self.raw_features = list(self.bundle["raw_features"])
        self.physics_features = list(self.bundle["physics_features"])
        self.hybrid_features = list(self.bundle["hybrid_features"])
        self.lower_bound = float(self.bundle.get("physical_lower_bound", 0.0))
        self.conformal = self.bundle.get("conformal_objects", {})
        self.ad = self.bundle.get("applicability_domain", {})
        self.coverage = float(self.conformal.get("coverage", 0.90))
        self.training_data = self.bundle.get("training_reference_data")
        self.molar = self.bundle.get(
            "molar_masses",
            {
                "NaOH": 39.997,
                "Na2O": 61.9789,
                "SiO2": 60.0843,
                "H2O": 18.01528,
                "Al2O3": 101.9613,
                "CaO": 56.0774,
            },
        )
        expected_raw = 24
        expected_hybrid = 38
        if len(self.raw_features) != expected_raw or len(self.hybrid_features) != expected_hybrid:
            raise ValueError(
                "This GUI requires the binder-revised 38-feature deployment bundle "
                f"({expected_raw} raw + 14 engineered predictors). The selected bundle contains "
                f"{len(self.raw_features)} raw and {len(self.hybrid_features)} total predictors."
            )
        required_binders = {
            "OPC (kg/m3)", "FA (kg/m3)", "GGBFS (kg/m3)",
            "SF (kg/m3)", "MK (kg/m3)", "Other SCM (kg/m3)",
        }
        if not required_binders.issubset(self.raw_features):
            raise ValueError("The selected bundle does not contain the revised binder variables.")

    @staticmethod
    def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    def engineer_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Reproduce all 14 physics-informed features from the final notebook."""
        data = raw.copy()
        for feature in self.raw_features:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")

        required = [
            "SiO2", "Al2O3", "CaO", "SiO2 (l)", "Na2O (l)",
            "Total NaOH (kg in 1m3 mix)",
            "Total Na2SiO3 (kg in 1m3 of mix)", "NaOH (Dry)",
            "Total water (in solutions + additional) (kg in 1m3 mix)",
            "Fine aggregate (kg in 1m3 mix)", "Coarse aggregate (kg/m3)",
            "Initial curing temp (C)", "Initial curing time (day)",
            "OPC (kg/m3)", "FA (kg/m3)", "GGBFS (kg/m3)",
            "SF (kg/m3)", "MK (kg/m3)", "Other SCM (kg/m3)",
        ]
        missing = [name for name in required if name not in data.columns]
        if missing:
            raise ValueError("Missing columns required for feature engineering:\n" + "\n".join(missing))

        mw_naoh = float(self.molar["NaOH"])
        mw_na2o = float(self.molar["Na2O"])
        mw_sio2 = float(self.molar["SiO2"])
        mw_h2o = float(self.molar["H2O"])
        mw_al2o3 = float(self.molar["Al2O3"])
        mw_cao = float(self.molar["CaO"])

        mol_si_bulk = data["SiO2"] / mw_sio2
        mol_al_bulk = 2.0 * data["Al2O3"] / mw_al2o3
        mol_ca_bulk = data["CaO"] / mw_cao
        data["PI_Si_Al"] = self._safe_div(mol_si_bulk, mol_al_bulk)
        data["PI_Ca_SiAl"] = self._safe_div(mol_ca_bulk, mol_si_bulk + mol_al_bulk)

        h2o_liq_fraction = (1.0 - data["Na2O (l)"] - data["SiO2 (l)"]).clip(0.0, 1.0)
        na2o_dry = data["Total Na2SiO3 (kg in 1m3 of mix)"] * data["Na2O (l)"]
        sio2_dry = data["Total Na2SiO3 (kg in 1m3 of mix)"] * data["SiO2 (l)"]
        mol_sio2_dry = sio2_dry / mw_sio2
        mol_sio2_liq = data["SiO2 (l)"] / mw_sio2
        mol_na2o_liq = data["Na2O (l)"] / mw_na2o
        data["PI_Silicate_Modulus"] = self._safe_div(mol_sio2_liq, mol_na2o_liq)
        data["PI_H2O_Na2O"] = self._safe_div(h2o_liq_fraction / mw_h2o, mol_na2o_liq)
        data["PI_NaOH_SS"] = self._safe_div(
            data["Total NaOH (kg in 1m3 mix)"],
            data["Total Na2SiO3 (kg in 1m3 of mix)"],
        )
        na2o_equivalent_from_naoh = data["NaOH (Dry)"] * mw_na2o / (2.0 * mw_naoh)
        total_effective_na2o = na2o_dry.fillna(0) + na2o_equivalent_from_naoh.fillna(0)
        mol_na_effective = 2.0 * total_effective_na2o / mw_na2o
        mol_si_effective = mol_si_bulk + mol_sio2_dry.fillna(0)
        data["PI_Na_Al_Effective"] = self._safe_div(mol_na_effective, mol_al_bulk)
        data["PI_Na_Si_Effective"] = self._safe_div(mol_na_effective, mol_si_effective)

        total_activator = (
            data["Total NaOH (kg in 1m3 mix)"].fillna(0)
            + data["Total Na2SiO3 (kg in 1m3 of mix)"].fillna(0)
        )
        tc_kelvin = data["Initial curing temp (C)"] + 273.15
        curing_days = data["Initial curing time (day)"].clip(lower=0)
        data["PI_Maturity"] = curing_days * np.exp(
            (40000.0 / 8.314) * (1.0 / 293.15 - 1.0 / tc_kelvin)
        )

        binder_columns = [
            "OPC (kg/m3)", "FA (kg/m3)", "GGBFS (kg/m3)",
            "SF (kg/m3)", "MK (kg/m3)", "Other SCM (kg/m3)",
        ]
        total_binder = data[binder_columns].fillna(0.0).sum(axis=1)
        total_aggregate = (
            data["Fine aggregate (kg in 1m3 mix)"].fillna(0.0)
            + data["Coarse aggregate (kg/m3)"].fillna(0.0)
        )
        data["PI_Activator_Binder"] = self._safe_div(total_activator, total_binder)
        data["PI_TotalWater_Binder"] = self._safe_div(
            data["Total water (in solutions + additional) (kg in 1m3 mix)"], total_binder
        )
        data["PI_Aggregate_Binder"] = self._safe_div(total_aggregate, total_binder)
        data["PI_EffectiveNa2O_Binder"] = self._safe_div(total_effective_na2o, total_binder)
        data["PI_DrySiO2_Binder"] = self._safe_div(sio2_dry, total_binder)
        data["PI_Superplasticizer_Binder"] = self._safe_div(
            data["Superplasticizer (kg in 1m3 mix)"], total_binder
        )
        return data

    def validate_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.raw_features if f not in raw.columns]
        if missing:
            raise ValueError("The input file is missing required columns:\n" + "\n".join(missing))
        clean = raw[self.raw_features].copy()
        for col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")
        fully_empty = clean.isna().all(axis=1)
        if fully_empty.any():
            clean = clean.loc[~fully_empty].reset_index(drop=True)
        if clean.empty:
            raise ValueError("No usable input rows were found.")
        if clean.isna().any(axis=None):
            missing_columns = clean.columns[clean.isna().any()].tolist()
            raise ValueError(
                "All 24 raw inputs are required. Missing or nonnumeric values were found in:\n"
                + "\n".join(missing_columns)
            )
        if (clean < 0).any(axis=None):
            negative_columns = clean.columns[(clean < 0).any()].tolist()
            raise ValueError("Negative input values are not permitted:\n" + "\n".join(negative_columns))
        if (clean["Age_days"] <= 0).any():
            raise ValueError("Testing age must be greater than zero days.")

        binder_columns = [
            "OPC (kg/m3)", "FA (kg/m3)", "GGBFS (kg/m3)",
            "SF (kg/m3)", "MK (kg/m3)", "Other SCM (kg/m3)",
        ]
        if (clean[binder_columns].sum(axis=1) <= 0).any():
            raise ValueError("The sum of the six binder constituents must be greater than zero.")
        liquid_fraction_sum = clean["Na2O (l)"] + clean["SiO2 (l)"]
        if (liquid_fraction_sum > 1.0 + 1e-9).any():
            raise ValueError("Na₂O and SiO₂ fractions in sodium-silicate solution cannot sum to more than 1.0.")
        if (clean["NaOH (Dry)"] > clean["Total NaOH (kg in 1m3 mix)"] + 1e-9).any():
            raise ValueError("Dry NaOH content cannot exceed the sodium-hydroxide solution mass.")

        # Coupled composition constraint: the four reported major oxides are
        # percentages and cannot collectively exceed 100 wt.%.
        complete_oxide_rows = clean[OXIDE_FEATURES].notna().all(axis=1)
        oxide_sum = clean[OXIDE_FEATURES].sum(axis=1, min_count=len(OXIDE_FEATURES))
        invalid_oxide_rows = complete_oxide_rows & (oxide_sum > OXIDE_SUM_MAX + 1e-9)
        if invalid_oxide_rows.any():
            row_numbers = (np.flatnonzero(invalid_oxide_rows.to_numpy()) + 2).tolist()
            displayed_rows = ", ".join(map(str, row_numbers[:10]))
            if len(row_numbers) > 10:
                displayed_rows += f", and {len(row_numbers) - 10} more"
            maximum_sum = float(oxide_sum.loc[invalid_oxide_rows].max())
            raise ValueError(
                "Invalid oxide composition: SiO₂ + Al₂O₃ + Fe₂O₃ + CaO "
                f"cannot exceed {OXIDE_SUM_MAX:.0f}%.\n"
                f"Affected spreadsheet row(s): {displayed_rows}.\n"
                f"Maximum detected sum: {maximum_sum:.3f}%."
            )
        return clean

    def _adaptive_scale(self, frame: pd.DataFrame) -> np.ndarray:
        scale_model = self.conformal["adaptive_scale_model"]
        raw = np.expm1(np.asarray(scale_model.predict(frame), dtype=float))
        global_scale = float(self.conformal["adaptive_global_error_scale"])
        floor = float(self.conformal["adaptive_scale_floor"])
        ceiling = float(self.conformal["adaptive_scale_ceiling"])
        shrinkage = float(self.conformal["adaptive_shrinkage"])
        raw = np.nan_to_num(raw, nan=global_scale, posinf=ceiling, neginf=floor)
        shrunk = (1.0 - shrinkage) * raw + shrinkage * global_scale
        return np.clip(shrunk, floor, ceiling)

    def _intervals(self, frame: pd.DataFrame, method: str):
        if method == "Split CP":
            model = self.conformal["split_model"]
            point = physical_predict(model, frame, self.lower_bound)
            q = float(self.conformal["split_q"])
            lower, upper = point - q, point + q
        elif method == "Stabilized Adaptive CP":
            model = self.conformal["adaptive_point_model"]
            point = physical_predict(model, frame, self.lower_bound)
            scale = self._adaptive_scale(frame)
            half_width = float(self.conformal["adaptive_q"]) * scale
            lower, upper = point - half_width, point + half_width
        elif method == "CV+":
            models = self.conformal["cvplus_models"]
            residual_groups = self.conformal["cvplus_residual_groups"]
            if not models or not residual_groups:
                raise ValueError("The deployment bundle does not contain CV+ models/residuals.")
            fold_predictions = np.vstack(
                [physical_predict(model, frame, self.lower_bound) for model in models]
            )
            point = fold_predictions.mean(axis=0)
            lower_parts, upper_parts = [], []
            for fold_pred, residuals in zip(fold_predictions, residual_groups):
                residuals = np.asarray(residuals, dtype=float).reshape(-1)
                lower_parts.append(fold_pred[:, None] - residuals[None, :])
                upper_parts.append(fold_pred[:, None] + residuals[None, :])
            lower_candidates = np.concatenate(lower_parts, axis=1)
            upper_candidates = np.concatenate(upper_parts, axis=1)
            ql, qu = cvplus_quantile_levels(lower_candidates.shape[1], 1.0 - self.coverage)
            lower = empirical_quantile(lower_candidates, ql, axis=1, method="lower")
            upper = empirical_quantile(upper_candidates, qu, axis=1, method="higher")
        else:
            raise ValueError(f"Unsupported conformal method: {method}")
        return point, np.maximum(lower, self.lower_bound), np.maximum(upper, self.lower_bound)

    def _domain_status(self, frame: pd.DataFrame):
        if not self.ad or not all(k in self.ad for k in ("imputer", "scaler", "nearest_neighbors", "threshold")):
            n = len(frame)
            return np.repeat("Not available", n), np.full(n, np.nan)
        transformed = self.ad["scaler"].transform(self.ad["imputer"].transform(frame))
        distance, _ = self.ad["nearest_neighbors"].kneighbors(transformed)
        distance = distance[:, 0]
        threshold = float(self.ad["threshold"])
        status = np.where(distance <= threshold, "In-domain", "Outside-domain")
        return status, distance

    def feature_ranges(self) -> dict[str, dict[str, float]]:
        """Return model-supported (5th–95th) and observed training ranges."""
        if not isinstance(self.training_data, pd.DataFrame):
            return {}
        ranges = {}
        for feature in self.raw_features:
            values = pd.to_numeric(self.training_data[feature], errors="coerce").dropna()
            if values.empty:
                continue
            ranges[feature] = {
                "supported_low": float(values.quantile(0.05)),
                "supported_high": float(values.quantile(0.95)),
                "observed_low": float(values.min()),
                "observed_high": float(values.max()),
            }
        return ranges

    def range_status(self, feature: str, value: float) -> str:
        limits = self.feature_ranges().get(feature)
        if limits is None or not np.isfinite(value):
            return "missing"
        if value < limits["observed_low"] or value > limits["observed_high"]:
            return "outside"
        if value < limits["supported_low"] or value > limits["supported_high"]:
            return "uncommon"
        return "supported"

    def range_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in frame[self.raw_features].iterrows():
            statuses = [self.range_status(feature, float(row[feature])) for feature in self.raw_features]
            rows.append({
                "Inputs_Supported": statuses.count("supported"),
                "Inputs_Uncommon": statuses.count("uncommon"),
                "Inputs_Outside_Observed": statuses.count("outside"),
                "Inputs_Missing": statuses.count("missing"),
            })
        return pd.DataFrame(rows)

    def predict(self, raw: pd.DataFrame, method: str) -> pd.DataFrame:
        clean = self.validate_raw(raw)
        engineered = self.engineer_features(clean)
        frame = engineered[self.hybrid_features]
        point, lower, upper = self._intervals(frame, method)
        status, distance = self._domain_status(frame)
        result = raw.loc[clean.index].reset_index(drop=True).copy()
        result["Predicted_Strength_MPa"] = point
        result["Lower_Bound_MPa"] = lower
        result["Upper_Bound_MPa"] = upper
        result["Interval_Width_MPa"] = upper - lower
        result["Nominal_Coverage"] = self.coverage
        result["Conformal_Method"] = method
        result["Applicability_Status"] = status
        result["Applicability_Distance"] = distance
        result["Applicability_Threshold"] = float(self.ad.get("threshold", np.nan))
        result["Major_Oxide_Sum_wt_pct"] = clean[OXIDE_FEATURES].sum(
            axis=1, min_count=len(OXIDE_FEATURES)
        ).to_numpy()
        result = pd.concat([result, self.range_summary(clean)], axis=1)
        return result

    def default_values(self) -> dict[str, float]:
        if isinstance(self.training_data, pd.DataFrame):
            # Use an actual representative training observation rather than an
            # artificial vector assembled from independent marginal medians.
            numeric = self.training_data[self.raw_features].apply(pd.to_numeric, errors="coerce")
            medians = numeric.median()
            filled = numeric.fillna(medians)
            iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
            scale = iqr.where(iqr > 1e-12, numeric.std()).replace(0, 1).fillna(1)
            distance = (((filled - medians) / scale) ** 2).mean(axis=1)
            representative = filled.loc[distance.idxmin()]
            return {f: float(representative[f]) for f in self.raw_features}
        return {f: 0.0 for f in self.raw_features}


class GeopolymerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.title(APP_TITLE)
        self.geometry("1460x860")
        self.minsize(1280, 760)
        self.backend: PredictionBackend | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.range_labels: dict[str, ctk.CTkLabel] = {}
        self.batch_input: pd.DataFrame | None = None
        self.batch_results: pd.DataFrame | None = None

        self._add_background()
        self._build_header()
        self._load_default_bundle()

    def _add_background(self):
        """Load a cropped banner image for the visible application header."""
        self._background_image = None
        path = resource_path(BACKGROUND_NAME)
        if path.exists():
            try:
                source = Image.open(path).convert("RGB")
                banner_size = (1416, 82)
                banner = ImageOps.fit(
                    source,
                    banner_size,
                    method=Image.Resampling.LANCZOS,
                    centering=(0.52, 0.46),
                )
                banner = banner.point(lambda value: int(value * 0.66))
                self._background_image = ctk.CTkImage(
                    light_image=banner, dark_image=banner, size=banner_size
                )
            except Exception:
                self._background_image = None

    def _build_header(self):
        header = ctk.CTkFrame(
            self, height=82, corner_radius=18,
            fg_color=("#DCE8F3", "#0B172A")
        )
        header.pack(fill="x", padx=22, pady=(14, 8))
        header.pack_propagate(False)

        if self._background_image is not None:
            # Rendering the title on the image label avoids opaque sibling
            # widgets concealing the photograph.
            title_surface = ctk.CTkLabel(
                header,
                image=self._background_image,
                text=f"   {APP_TITLE}",
                compound="center",
                anchor="w",
                font=ctk.CTkFont(size=23, weight="bold"),
                text_color="#F8FAFC",
                corner_radius=18,
            )
            title_surface.place(x=0, y=0, relwidth=1, relheight=1)
            controls_parent = title_surface
        else:
            title_surface = ctk.CTkLabel(
                header, text=APP_TITLE, anchor="w",
                font=ctk.CTkFont(size=23, weight="bold")
            )
            title_surface.pack(fill="both", expand=True, padx=20)
            controls_parent = header

        self.bundle_status = ctk.CTkLabel(
            controls_parent, text="Loading model…", text_color="#FBBF24",
            fg_color="#091629", corner_radius=8, width=104, height=30
        )
        self.bundle_status.place(relx=0.985, rely=0.5, anchor="e")
        ctk.CTkButton(
            controls_parent, text="Select Model Bundle", command=self._choose_bundle,
            width=150, height=34
        ).place(relx=0.895, rely=0.5, anchor="e")

    def _load_default_bundle(self):
        path = resource_path(BUNDLE_NAME)
        if not path.exists():
            self.bundle_status.configure(text="Bundle not found", text_color="#F87171")
            self._show_bundle_help()
            return
        self._load_bundle(path)

    def _choose_bundle(self):
        path = filedialog.askopenfilename(
            title="Select deployment bundle", filetypes=[("Joblib bundle", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self._load_bundle(Path(path))

    def _load_bundle(self, path: Path):
        try:
            self.backend = PredictionBackend(path)
            self.bundle_status.configure(text="Model ready", text_color="#34D399")
            self._build_workspace()
        except Exception as exc:
            self.backend = None
            self.bundle_status.configure(text="Model load failed", text_color="#F87171")
            messagebox.showerror(
                "Could not load model",
                f"{exc}\n\nUse the package versions in requirements.txt. "
                "If the problem persists, re-export the bundle from the final modeling notebook."
            )

    def _show_bundle_help(self):
        messagebox.showinfo(
            "Model bundle required",
            f"Place {BUNDLE_NAME} beside this application or click 'Select Model Bundle'.",
        )

    def _build_workspace(self):
        if hasattr(self, "tabs"):
            self.tabs.destroy()
        self.tabs = ctk.CTkTabview(self, corner_radius=18)
        self.tabs.pack(fill="both", expand=True, padx=22, pady=(4, 20))
        manual_tab = self.tabs.add("Manual Prediction")
        batch_tab = self.tabs.add("Batch Prediction")
        self._build_manual_tab(manual_tab)
        self._build_batch_tab(batch_tab)

    def _build_manual_tab(self, tab):
        # Two compact input columns keep all 24 variables visible without scrolling.
        left = ctk.CTkFrame(tab, width=960, corner_radius=16)
        left.pack(side="left", fill="both", expand=True, padx=(6, 8), pady=6)
        right = ctk.CTkFrame(tab, width=380, corner_radius=16)
        right.pack(side="right", fill="both", padx=(8, 6), pady=6)
        right.pack_propagate(False)

        defaults = self.backend.default_values()
        ranges = self.backend.feature_ranges()
        self.entries = {}
        self.range_labels = {}
        ctk.CTkLabel(
            left, text="MIXTURE AND CURING INPUTS", anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#60A5FA"
        ).pack(fill="x", padx=14, pady=(9, 3))

        columns = ctk.CTkFrame(left, fg_color="transparent")
        columns.pack(fill="both", expand=True, padx=8, pady=2)
        column_frames = []
        for column_index in range(2):
            frame = ctk.CTkFrame(columns, fg_color="transparent")
            frame.grid(row=0, column=column_index, sticky="nsew", padx=6)
            frame.grid_columnconfigure(0, weight=1)
            column_frames.append(frame)
            columns.grid_columnconfigure(column_index, weight=1, uniform="input_columns")
        columns.grid_rowconfigure(0, weight=1)

        split_index = math.ceil(len(self.backend.raw_features) / 2)
        feature_columns = [
            self.backend.raw_features[:split_index],
            self.backend.raw_features[split_index:],
        ]

        for column_index, features in enumerate(feature_columns):
            frame = column_frames[column_index]
            current_group = None
            row = 0
            for feature in features:
                group = feature_group(feature)
                if group != current_group:
                    current_group = group
                    group_text = group
                    if column_index == 1 and row == 0 and group == feature_group(feature_columns[0][-1]):
                        group_text += " (continued)"
                    ctk.CTkLabel(
                        frame, text=group_text.upper(), anchor="w",
                        font=ctk.CTkFont(size=10, weight="bold"), text_color="#60A5FA"
                    ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=4, pady=(5, 1))
                    row += 1

                label = ctk.CTkLabel(
                    frame, text=FRIENDLY_LABELS.get(feature, feature), anchor="w",
                    wraplength=215, font=ctk.CTkFont(size=11)
                )
                label.grid(row=row, column=0, sticky="w", padx=(4, 5), pady=2)

                entry = ctk.CTkEntry(
                    frame, width=94, height=25, font=ctk.CTkFont(size=11), border_width=1
                )
                entry.insert(0, f"{defaults.get(feature, 0.0):.6g}")
                entry.grid(row=row, column=1, padx=4, pady=2)

                limits = ranges.get(feature, {})
                range_text = (
                    f"5–95%: {compact_number(limits.get('supported_low', np.nan))}–"
                    f"{compact_number(limits.get('supported_high', np.nan))}\n"
                    f"Full: {compact_number(limits.get('observed_low', np.nan))}–"
                    f"{compact_number(limits.get('observed_high', np.nan))}"
                )
                range_label = ctk.CTkLabel(
                    frame, text=range_text, anchor="w", justify="left", width=125,
                    font=ctk.CTkFont(size=9), text_color="#94A3B8"
                )
                range_label.grid(row=row, column=2, sticky="w", padx=(4, 2), pady=1)

                self.entries[feature] = entry
                self.range_labels[feature] = range_label
                entry.bind("<KeyRelease>", lambda _event: self._update_input_guidance())
                entry.bind("<FocusOut>", lambda _event: self._update_input_guidance())
                row += 1

        controls = ctk.CTkFrame(left, fg_color="transparent", height=34)
        controls.pack(fill="x", padx=14, pady=(2, 7))
        ctk.CTkButton(
            controls, text="Load representative mixture", width=170, height=28,
            font=ctk.CTkFont(size=11), command=self._reset_inputs
        ).pack(side="left", padx=(0, 8))
        self.input_summary = ctk.CTkLabel(
            controls, text="", anchor="w", font=ctk.CTkFont(size=10)
        )
        self.input_summary.pack(side="left", padx=5)
        self.oxide_sum_label = ctk.CTkLabel(
            controls, text="Major oxide sum: —", anchor="e",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.oxide_sum_label.pack(side="right", padx=5)
        self._update_input_guidance()

        ctk.CTkLabel(right, text="Prediction", font=ctk.CTkFont(size=21, weight="bold")).pack(pady=(17, 6))
        self.manual_method = ctk.CTkOptionMenu(
            right, values=["Stabilized Adaptive CP", "Split CP", "CV+"], width=260
        )
        self.manual_method.pack(pady=5)
        self.predict_button = ctk.CTkButton(
            right, text="Predict Strength", height=38, command=self._manual_predict
        )
        self.predict_button.pack(pady=8)
        self._update_input_guidance()
        self.point_label = self._result_card(right, "Point estimate", "— MPa", "#38BDF8")
        self.lower_label = self._result_card(right, "Lower bound", "— MPa", "#34D399")
        self.upper_label = self._result_card(right, "Upper bound", "— MPa", "#FBBF24")
        self.interval_label = ctk.CTkLabel(
            right, text="Prediction interval: —", wraplength=350,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.interval_label.pack(pady=(10, 3))
        self.domain_label = ctk.CTkLabel(right, text="Applicability: —", wraplength=350)
        self.domain_label.pack(pady=(7, 4))
        self.range_summary_label = ctk.CTkLabel(right, text="Input ranges: —", wraplength=350)
        self.range_summary_label.pack(pady=3)
        ctk.CTkLabel(
            right,
            text="Ranges describe model support, not optimal mixture design.\nPrediction intervals use 90% nominal coverage.",
            text_color="#94A3B8", wraplength=350
        ).pack(pady=(5, 3))

    @staticmethod
    def _result_card(parent, title, value, color):
        frame = ctk.CTkFrame(parent, corner_radius=14, fg_color=("#E2E8F0", "#16243A"))
        frame.pack(fill="x", padx=24, pady=5)
        ctk.CTkLabel(
            frame, text=title, text_color="#94A3B8", font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=14, pady=(7, 0))
        label = ctk.CTkLabel(
            frame, text=value, font=ctk.CTkFont(size=21, weight="bold"), text_color=color
        )
        label.pack(anchor="w", padx=14, pady=(0, 7))
        return label

    def _manual_predict(self):
        try:
            values = {}
            for feature, entry in self.entries.items():
                text = entry.get().strip()
                if not text:
                    raise ValueError(f"Enter a value for {FRIENDLY_LABELS.get(feature, feature)}.")
                else:
                    values[feature] = float(text)
            result = self.backend.predict(pd.DataFrame([values]), self.manual_method.get()).iloc[0]
            self.point_label.configure(text=f"{result['Predicted_Strength_MPa']:.2f} MPa")
            self.lower_label.configure(text=f"{result['Lower_Bound_MPa']:.2f} MPa")
            self.upper_label.configure(text=f"{result['Upper_Bound_MPa']:.2f} MPa")
            self.interval_label.configure(
                text=(
                    f"{result['Lower_Bound_MPa']:.2f}  ├──── ● ────┤  "
                    f"{result['Upper_Bound_MPa']:.2f} MPa\n"
                    f"Interval width: {result['Interval_Width_MPa']:.2f} MPa"
                ),
                text_color="#CBD5E1",
            )
            outside = result["Applicability_Status"] == "Outside-domain"
            distance = result["Applicability_Distance"]
            threshold = result["Applicability_Threshold"]
            self.domain_label.configure(
                text=(
                    f"⚠ Outside applicability domain\nDistance {distance:.3f} > threshold {threshold:.3f}; interpret cautiously"
                    if outside else
                    f"✓ Within applicability domain\nDistance {distance:.3f} ≤ threshold {threshold:.3f}"
                ),
                text_color=("#FBBF24" if outside else "#34D399"),
            )
            uncommon = int(result["Inputs_Uncommon"])
            out = int(result["Inputs_Outside_Observed"])
            supported = int(result["Inputs_Supported"])
            self.range_summary_label.configure(
                text=f"Input ranges: {supported} supported • {uncommon} uncommon • {out} outside observed",
                text_color=("#F87171" if out else "#FBBF24" if uncommon else "#34D399"),
            )
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))

    def _reset_inputs(self):
        defaults = self.backend.default_values()
        for feature, entry in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, f"{defaults.get(feature, 0.0):.6g}")
        self._update_input_guidance()

    def _update_input_guidance(self):
        counts = {"supported": 0, "uncommon": 0, "outside": 0, "missing": 0}
        colors = {
            "supported": "#34D399", "uncommon": "#FBBF24",
            "outside": "#F87171", "missing": "#94A3B8",
        }
        for feature, entry in self.entries.items():
            try:
                value = float(entry.get().strip())
                status = self.backend.range_status(feature, value)
            except Exception:
                status = "missing"
            counts[status] += 1
            entry.configure(border_color=colors[status])
            self.range_labels[feature].configure(text_color=colors[status])
        if hasattr(self, "input_summary"):
            self.input_summary.configure(
                text=(
                    f"{counts['supported']} supported  •  {counts['uncommon']} uncommon  •  "
                    f"{counts['outside']} outside  •  {counts['missing']} missing"
                ),
                text_color=(
                    "#F87171" if counts["outside"] or counts["missing"]
                    else "#FBBF24" if counts["uncommon"] else "#34D399"
                ),
            )

        # Live coupled validation for the major-oxide percentage balance.
        oxide_values = []
        for feature in OXIDE_FEATURES:
            try:
                oxide_values.append(float(self.entries[feature].get().strip()))
            except Exception:
                oxide_values = []
                break

        oxide_valid = bool(oxide_values) and all(np.isfinite(oxide_values))
        oxide_sum = float(sum(oxide_values)) if oxide_valid else np.nan
        exceeds_limit = oxide_valid and oxide_sum > OXIDE_SUM_MAX + 1e-9

        if hasattr(self, "oxide_sum_label"):
            if not oxide_valid:
                self.oxide_sum_label.configure(
                    text="Major oxide sum: incomplete", text_color="#F87171"
                )
            elif exceeds_limit:
                self.oxide_sum_label.configure(
                    text=f"Major oxide sum: {oxide_sum:.2f}% > 100%",
                    text_color="#F87171",
                )
            else:
                self.oxide_sum_label.configure(
                    text=f"Major oxide sum: {oxide_sum:.2f}% ≤ 100%",
                    text_color="#34D399",
                )

        for feature in OXIDE_FEATURES:
            if exceeds_limit:
                self.entries[feature].configure(border_color="#F87171")
                self.range_labels[feature].configure(text_color="#F87171")

        if hasattr(self, "predict_button"):
            self.predict_button.configure(
                state="normal" if oxide_valid and not exceeds_limit else "disabled"
            )

    def _build_batch_tab(self, tab):
        panel = ctk.CTkFrame(tab, corner_radius=16)
        panel.pack(fill="both", expand=True, padx=12, pady=12)
        ctk.CTkLabel(panel, text="Batch Excel/CSV Prediction", font=ctk.CTkFont(size=25, weight="bold")).pack(pady=(35, 8))
        ctk.CTkLabel(
            panel,
            text="Upload a file containing the 24 raw predictor columns. All 14 physics-informed features are calculated automatically.",
            wraplength=760,
        ).pack(pady=8)
        self.batch_method = ctk.CTkOptionMenu(panel, values=["Stabilized Adaptive CP", "Split CP", "CV+"], width=280)
        self.batch_method.pack(pady=12)
        buttons = ctk.CTkFrame(panel, fg_color="transparent")
        buttons.pack(pady=15)
        ctk.CTkButton(buttons, text="Download Input Template", command=self._save_template, width=190).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Upload Excel or CSV", command=self._load_batch, width=180).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Run Batch Prediction", command=self._run_batch, width=190).pack(side="left", padx=8)
        ctk.CTkButton(buttons, text="Export Results", command=self._export_batch, width=160).pack(side="left", padx=8)
        self.batch_status = ctk.CTkLabel(panel, text="No file selected", text_color="#94A3B8")
        self.batch_status.pack(pady=14)
        self.batch_preview = ctk.CTkTextbox(panel, height=330, font=("Courier New", 12))
        self.batch_preview.pack(fill="both", expand=True, padx=25, pady=(5, 25))

    def _save_template(self):
        path = filedialog.asksaveasfilename(
            title="Save input template", defaultextension=".xlsx", initialfile="geopolymer_input_template.xlsx",
            filetypes=[("Excel workbook", "*.xlsx")]
        )
        if path:
            ranges = self.backend.feature_ranges()
            guide = pd.DataFrame([
                {
                    "Required_Column_Name": feature,
                    "Display_Name": FRIENDLY_LABELS.get(feature, feature),
                    "Category": feature_group(feature),
                    "Model_Supported_Low_5th": ranges.get(feature, {}).get("supported_low", np.nan),
                    "Model_Supported_High_95th": ranges.get(feature, {}).get("supported_high", np.nan),
                    "Observed_Minimum": ranges.get(feature, {}).get("observed_low", np.nan),
                    "Observed_Maximum": ranges.get(feature, {}).get("observed_high", np.nan),
                }
                for feature in self.backend.raw_features
            ])
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame(columns=self.backend.raw_features).to_excel(
                    writer, sheet_name="Input_Template", index=False
                )
                guide.to_excel(writer, sheet_name="Variable_Guide", index=False)
            self.batch_status.configure(text=f"Template saved: {Path(path).name}", text_color="#34D399")

    def _load_batch(self):
        path = filedialog.askopenfilename(
            title="Select input file", filetypes=[("Excel or CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.batch_input = pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_excel(path)
            self.batch_results = None
            self.batch_status.configure(text=f"Loaded {len(self.batch_input)} rows from {Path(path).name}", text_color="#38BDF8")
            self._preview(self.batch_input)
        except Exception as exc:
            messagebox.showerror("File error", str(exc))

    def _run_batch(self):
        if self.batch_input is None:
            messagebox.showwarning("No input file", "Upload an Excel or CSV file first.")
            return
        try:
            self.batch_results = self.backend.predict(self.batch_input, self.batch_method.get())
            outside = int((self.batch_results["Applicability_Status"] == "Outside-domain").sum())
            range_outside = int((self.batch_results["Inputs_Outside_Observed"] > 0).sum())
            range_uncommon = int((self.batch_results["Inputs_Uncommon"] > 0).sum())
            self.batch_status.configure(
                text=(
                    f"Completed {len(self.batch_results)} predictions • {outside} outside-domain • "
                    f"{range_outside} with out-of-range inputs • {range_uncommon} with uncommon inputs"
                ),
                text_color=("#F87171" if range_outside else "#FBBF24" if outside or range_uncommon else "#34D399")
            )
            self._preview(self.batch_results)
        except Exception as exc:
            messagebox.showerror("Batch prediction error", str(exc))

    def _export_batch(self):
        if self.batch_results is None:
            messagebox.showwarning("No results", "Run batch prediction before exporting.")
            return
        path = filedialog.asksaveasfilename(
            title="Export prediction results", defaultextension=".xlsx", initialfile="geopolymer_predictions.xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("CSV file", "*.csv")]
        )
        if path:
            if Path(path).suffix.lower() == ".csv":
                self.batch_results.to_csv(path, index=False)
            else:
                self.batch_results.to_excel(path, index=False)
            self.batch_status.configure(text=f"Results saved: {Path(path).name}", text_color="#34D399")

    def _preview(self, frame: pd.DataFrame):
        self.batch_preview.delete("1.0", "end")
        with pd.option_context("display.max_columns", 10, "display.width", 150):
            self.batch_preview.insert("1.0", frame.head(15).to_string(index=False))


def main():
    try:
        app = GeopolymerApp()
        app.mainloop()
    except Exception:
        error = traceback.format_exc()
        try:
            messagebox.showerror("Application error", error)
        except Exception:
            print(error)


if __name__ == "__main__":
    main()
