"""Inference for the trained species distribution model.

Loads a model bundle (defaults to Model2SDM/sdm_logreg.joblib) and scores
per-species presence probabilities from environmental + space/time features.

Programmatic:
    from inference import SDMInference
    m = SDMInference()                             # loads sdm_logreg.joblib by default
    # Single point:
    probs = m.predict_point(
        sst=15.2, sst_anomaly=0.3, chlorophyll=0.8,
        salinity=33.5, dissolved_oxygen=6.2, ssh=0.05,
        lat=36.5, lon=-122.0, day_of_year=172, month=6,
    )                                              # dict: species -> probability
    # DataFrame (must have the raw FEATURE_COLUMNS):
    df_out = m.predict(df)                         # rows x species probability matrix

CLI:
    python inference.py --csv input.csv --out probs.csv
    python inference.py --bundle sdm_logreg.joblib --csv input.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS
from data import build_feature_matrix

ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "sdm_logreg.joblib"


class SDMInference:
    def __init__(self, bundle_path: Optional[Path | str] = None) -> None:
        path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE
        if not path.exists():
            raise FileNotFoundError(f"model bundle not found: {path}")
        bundle = joblib.load(path)
        self.bundle_path = path
        self.model_kind = bundle.get("model_kind", "unknown")
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.species: list[str] = list(bundle["species"])
        self.estimators: dict = bundle["estimators"]

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"missing required feature columns: {missing}")
        X = build_feature_matrix(df[FEATURE_COLUMNS])
        extra = [c for c in self.feature_names if c not in X.columns]
        if extra:
            raise ValueError(f"feature matrix missing expected columns: {extra}")
        return X[self.feature_names].values

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return an (n_rows, n_species) probability DataFrame."""
        X = self._prepare(df)
        out = {}
        for species, est in self.estimators.items():
            out[species] = est.predict_proba(X)[:, 1].astype("float32")
        return pd.DataFrame(out, index=df.index)

    def predict_point(self, **features) -> dict[str, float]:
        """Score a single observation passed as keyword features."""
        row = pd.DataFrame([features])
        probs = self.predict(row).iloc[0]
        return {species: float(probs[species]) for species in self.species}

    def top_species(self, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Return the top-k species per row with their probabilities."""
        probs = self.predict(df)
        order = np.argsort(-probs.values, axis=1)[:, :k]
        species_arr = np.asarray(probs.columns)
        records = []
        for i, row_order in enumerate(order):
            for rank, j in enumerate(row_order):
                records.append({
                    "row": i,
                    "rank": rank,
                    "species": species_arr[j],
                    "probability": float(probs.values[i, j]),
                })
        return pd.DataFrame(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    ap.add_argument("--csv", required=True,
                    help=f"Input CSV with columns: {', '.join(FEATURE_COLUMNS)}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-k", type=int, default=None,
                    help="If set, emit long-format top-k species per row instead of wide probabilities")
    args = ap.parse_args()

    model = SDMInference(bundle_path=args.bundle)
    print(f"[inference] bundle={model.bundle_path.name}  "
          f"kind={model.model_kind}  n_species={len(model.species)}")

    df = pd.read_csv(args.csv)
    print(f"[inference] scoring {len(df):,} rows")

    if args.top_k is not None:
        out = model.top_species(df, k=args.top_k)
    else:
        out = model.predict(df)

    out_path = Path(args.out) if args.out else Path(args.csv).with_suffix(".probs.csv")
    out.to_csv(out_path, index=False)
    print(f"[inference] wrote {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
