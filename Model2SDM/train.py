"""Train species distribution models.

Fits a logistic regression (or gradient boosting) classifier per species
on the multi-label training_data.csv. Each classifier produces
P(species present | environmental features, location, time).

Usage:
    python train.py                  # logistic regression (default)
    python train.py --model gbm      # histogram gradient boosting
    python train.py --min-positives 50
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ARTIFACTS_DIR, MIN_POSITIVES, RANDOM_STATE, TEST_FRACTION
from data import build_feature_matrix, load_dataset, split_features_labels


def make_estimator(kind):
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=None,
            )),
        ])
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def train(model_kind="logreg", min_positives=MIN_POSITIVES):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading training data...")
    df = load_dataset()
    print(f"[train] rows={len(df):,}  cols={df.shape[1]:,}")

    X_raw, Y, species_list, pos_counts = split_features_labels(df, min_positives=min_positives)
    print(f"[train] species with >= {min_positives} positives: {len(species_list)}")

    X = build_feature_matrix(df.loc[X_raw.index])
    feature_names = list(X.columns)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_FRACTION, random_state=RANDOM_STATE, shuffle=True,
    )

    models = {}
    metrics = []

    for i, species in enumerate(species_list):
        y_train = Y_train[species].values
        y_test = Y_test[species].values

        if y_train.sum() < 2 or y_train.sum() == len(y_train):
            continue

        t0 = time.time()
        est = make_estimator(model_kind)
        est.fit(X_train.values, y_train)

        proba = est.predict_proba(X_test.values)[:, 1]
        y_pred = (proba >= 0.5).astype(np.int8)
        acc = accuracy_score(y_test, y_pred)
        baseline_acc = max((y_test == 0).mean(), (y_test == 1).mean())
        n_correct = int((y_pred == y_test).sum())
        n_total = int(len(y_test))

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            auc = float("nan")
            ap = float("nan")
        else:
            auc = roc_auc_score(y_test, proba)
            ap = average_precision_score(y_test, proba)

        models[species] = est
        metrics.append({
            "species": species,
            "n_positives_total": int(pos_counts[species]),
            "n_positives_train": int(y_train.sum()),
            "n_positives_test": int(y_test.sum()),
            "roc_auc": float(auc),
            "avg_precision": float(ap),
            "accuracy": float(acc),
            "baseline_accuracy": float(baseline_acc),
            "n_correct": n_correct,
            "n_total": n_total,
            "fit_seconds": round(time.time() - t0, 3),
        })

        if (i + 1) % 25 == 0 or i == len(species_list) - 1:
            valid_aucs = [m["roc_auc"] for m in metrics if not np.isnan(m["roc_auc"])]
            mean_auc = np.mean(valid_aucs) if valid_aucs else float("nan")
            mean_acc = np.mean([m["accuracy"] for m in metrics])
            print(f"[train] {i+1}/{len(species_list)} species fit  "
                  f"mean_auc={mean_auc:.3f}  mean_acc={mean_acc:.3f}")

    bundle_path = ARTIFACTS_DIR / f"sdm_{model_kind}.joblib"
    joblib.dump({
        "model_kind": model_kind,
        "feature_names": feature_names,
        "species": list(models.keys()),
        "estimators": models,
    }, bundle_path, compress=3)

    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(ARTIFACTS_DIR / f"metrics_{model_kind}.csv", index=False)

    total_correct = int(metrics_df["n_correct"].sum())
    total_predictions = int(metrics_df["n_total"].sum())
    micro_accuracy = total_correct / total_predictions if total_predictions else float("nan")

    summary = {
        "model_kind": model_kind,
        "n_species": len(models),
        "mean_roc_auc": float(metrics_df["roc_auc"].mean()),
        "median_roc_auc": float(metrics_df["roc_auc"].median()),
        "mean_avg_precision": float(metrics_df["avg_precision"].mean()),
        "mean_accuracy": float(metrics_df["accuracy"].mean()),
        "median_accuracy": float(metrics_df["accuracy"].median()),
        "mean_baseline_accuracy": float(metrics_df["baseline_accuracy"].mean()),
        "micro_accuracy": float(micro_accuracy),
        "total_correct": total_correct,
        "total_predictions": total_predictions,
        "feature_names": feature_names,
        "min_positives": min_positives,
    }
    with open(ARTIFACTS_DIR / f"summary_{model_kind}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[train] saved bundle -> {bundle_path}")
    print(f"[train] mean ROC AUC = {summary['mean_roc_auc']:.3f}  "
          f"median ROC AUC = {summary['median_roc_auc']:.3f}")
    print(f"[train] mean accuracy = {summary['mean_accuracy']:.4f}  "
          f"(baseline {summary['mean_baseline_accuracy']:.4f})")
    print(f"[train] micro accuracy = {summary['micro_accuracy']:.4f}  "
          f"({summary['total_correct']:,}/{summary['total_predictions']:,})")
    return bundle_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    p.add_argument("--min-positives", type=int, default=MIN_POSITIVES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(model_kind=args.model, min_positives=args.min_positives)
