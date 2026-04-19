from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path(
    r"C:\Users\peter\OneDrive\Desktop\Oceanpulse\Model2DataEngineering\outputs\training_data.csv"
)
OUTPUT_DIR = Path(
    r"C:\Users\peter\Documents\Codex\2026-04-19-files-mentioned-by-the-user-training\outputs"
)
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_POSITIVES_FOR_MODEL = 25


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        probs = np.full(len(x), self.probability, dtype=float)
        return np.column_stack([1.0 - probs, probs])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["time"] = pd.to_datetime(work["time"])
    doy_angle = 2.0 * np.pi * work["day_of_year"] / 365.25
    month_angle = 2.0 * np.pi * work["month"] / 12.0

    features = pd.DataFrame(
        {
            "lat": work["lat"].astype(float),
            "lon": work["lon"].astype(float),
            "sst": work["sst"].astype(float),
            "sst_anomaly": work["sst_anomaly"].astype(float),
            "chlorophyll_log1p": np.log1p(work["chlorophyll"].astype(float)),
            "salinity": work["salinity"].astype(float),
            "dissolved_oxygen": work["dissolved_oxygen"].astype(float),
            "ssh": work["ssh"].astype(float),
            "day_of_year_sin": np.sin(doy_angle),
            "day_of_year_cos": np.cos(doy_angle),
            "month_sin": np.sin(month_angle),
            "month_cos": np.cos(month_angle),
            "year": work["time"].dt.year.astype(float),
        }
    )
    return features


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {
        "positive_rate_test": float(y_true.mean()),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def train_all_species(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metadata_cols = list(df.columns[:11])
    species_cols = list(df.columns[11:])
    features = build_features(df[metadata_cols])

    x_train, x_test, idx_train, idx_test = train_test_split(
        features,
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    models: Dict[str, Pipeline | ConstantProbabilityModel] = {}
    metrics_rows: List[dict] = []

    for species in species_cols:
        y = df[species].astype(int)
        y_train = y.loc[idx_train]
        y_test = y.loc[idx_test]
        positive_count = int(y.sum())

        if positive_count < MIN_POSITIVES_FOR_MODEL or y_train.nunique() < 2:
            prevalence = float(y_train.mean())
            model: Pipeline | ConstantProbabilityModel = ConstantProbabilityModel(prevalence)
            model_type = "constant_prevalence_baseline"
        else:
            model = make_classifier()
            model.fit(x_train, y_train)
            model_type = "logistic_regression"

        y_prob = model.predict_proba(x_test)[:, 1]
        model_metrics = evaluate_predictions(y_test, y_prob)
        model_metrics.update(
            {
                "species": species,
                "positive_count_total": positive_count,
                "positive_count_train": int(y_train.sum()),
                "positive_count_test": int(y_test.sum()),
                "model_type": model_type,
            }
        )
        metrics_rows.append(model_metrics)
        models[species] = model

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["average_precision", "roc_auc", "positive_count_total"],
        ascending=[False, False, False],
        na_position="last",
    )

    return models, metrics_df, features


def fit_final_models(df: pd.DataFrame, features: pd.DataFrame) -> dict:
    species_cols = list(df.columns[11:])
    final_models: Dict[str, Pipeline | ConstantProbabilityModel] = {}

    for species in species_cols:
        y = df[species].astype(int)
        if int(y.sum()) < MIN_POSITIVES_FOR_MODEL or y.nunique() < 2:
            final_models[species] = ConstantProbabilityModel(float(y.mean()))
            continue

        model = make_classifier()
        model.fit(features, y)
        final_models[species] = model

    return final_models


def build_probability_maps(
    df: pd.DataFrame,
    features: pd.DataFrame,
    final_models: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = df.iloc[:, :11].copy()
    species_cols = list(df.columns[11:])
    probability_columns = {
        species: final_models[species].predict_proba(features)[:, 1] for species in species_cols
    }
    probability_frame = pd.DataFrame(probability_columns, index=df.index)

    location_probabilities = probability_frame.groupby([metadata["lat"], metadata["lon"]]).mean()
    location_observed = df[species_cols].groupby([metadata["lat"], metadata["lon"]]).mean()
    sample_count = metadata.groupby(["lat", "lon"]).size().rename("n_samples")

    map_df = (
        location_probabilities.stack()
        .rename("mean_predicted_probability")
        .reset_index()
        .rename(columns={"level_2": "species"})
        .merge(
            location_observed.stack().rename("observed_prevalence").reset_index().rename(
                columns={"level_2": "species"}
            ),
            on=["lat", "lon", "species"],
            how="left",
        )
        .merge(sample_count.reset_index(), on=["lat", "lon"], how="left")
        .sort_values(["species", "lat", "lon"])
    )

    row_level = pd.concat([metadata[["time", "lat", "lon"]], probability_frame], axis=1)
    return map_df, row_level


def plot_metric_distributions(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics_df["positive_count_total"].plot.hist(ax=axes[0], bins=30)
    axes[0].set_title("Positive Counts per Species")
    axes[0].set_xlabel("Positive examples")

    metrics_df["average_precision"].dropna().plot.hist(ax=axes[1], bins=30)
    axes[1].set_title("Average Precision")
    axes[1].set_xlabel("AP")

    metrics_df["roc_auc"].dropna().plot.hist(ax=axes[2], bins=30)
    axes[2].set_title("ROC AUC")
    axes[2].set_xlabel("ROC AUC")

    fig.tight_layout()
    fig.savefig(output_dir / "model_metric_distributions.png", dpi=150)
    plt.close(fig)


def write_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    logistic_mask = metrics_df["model_type"] == "logistic_regression"
    summary = {
        "n_species": int(len(metrics_df)),
        "n_logistic_models": int(logistic_mask.sum()),
        "n_baseline_models": int((~logistic_mask).sum()),
        "min_positives_for_logistic_model": MIN_POSITIVES_FOR_MODEL,
        "median_positive_count": float(metrics_df["positive_count_total"].median()),
        "median_average_precision_logistic": float(
            metrics_df.loc[logistic_mask, "average_precision"].dropna().median()
        )
        if logistic_mask.any()
        else None,
        "median_roc_auc_logistic": float(
            metrics_df.loc[logistic_mask, "roc_auc"].dropna().median()
        )
        if logistic_mask.any()
        else None,
    }
    (output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    models, metrics_df, features = train_all_species(df)
    final_models = fit_final_models(df, features)
    probability_maps_df, row_level_probs_df = build_probability_maps(df, features, final_models)

    metrics_df.to_csv(OUTPUT_DIR / "species_model_metrics.csv", index=False)
    probability_maps_df.to_csv(OUTPUT_DIR / "species_probability_maps.csv", index=False)
    row_level_probs_df.to_csv(
        OUTPUT_DIR / "species_row_level_probabilities.csv.gz",
        index=False,
        compression="gzip",
    )
    plot_metric_distributions(metrics_df, OUTPUT_DIR)
    write_summary(metrics_df, OUTPUT_DIR)

    bundle = {
        "data_path": str(DATA_PATH),
        "feature_columns": list(features.columns),
        "species_columns": list(df.columns[11:]),
        "models": final_models,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "min_positives_for_model": MIN_POSITIVES_FOR_MODEL,
    }
    joblib.dump(bundle, OUTPUT_DIR / "species_distribution_model.joblib")

    top_ap = metrics_df[["species", "model_type", "positive_count_total", "average_precision", "roc_auc"]].head(10)
    print("Saved outputs to:", OUTPUT_DIR)
    print("Top species by average precision:")
    print(top_ap.to_string(index=False))


if __name__ == "__main__":
    main()
matplotlib.use("Agg")
