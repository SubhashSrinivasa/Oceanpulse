import numpy as np
import pandas as pd

from config import (
    TRAINING_CSV,
    FEATURE_COLUMNS,
    NON_SPECIES_COLUMNS,
    MIN_POSITIVES,
)


def load_dataset():
    df = pd.read_csv(TRAINING_CSV)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def split_features_labels(df, min_positives=MIN_POSITIVES):
    species_cols = [c for c in df.columns if c not in NON_SPECIES_COLUMNS]

    X = df[FEATURE_COLUMNS].astype("float32").copy()
    for col in FEATURE_COLUMNS:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    Y = df[species_cols].astype("int8")

    positives_per_species = Y.sum(axis=0)
    keep = positives_per_species[positives_per_species >= min_positives].index.tolist()
    Y = Y[keep]

    return X, Y, keep, positives_per_species.loc[keep].to_dict()


def cyclical_time_encoding(day_of_year, month):
    doy = np.asarray(day_of_year, dtype=np.float32)
    mon = np.asarray(month, dtype=np.float32)
    return pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 366.0),
        "doy_cos": np.cos(2 * np.pi * doy / 366.0),
        "month_sin": np.sin(2 * np.pi * mon / 12.0),
        "month_cos": np.cos(2 * np.pi * mon / 12.0),
    })


def build_feature_matrix(df):
    base = df[FEATURE_COLUMNS].astype("float32").copy()
    for col in FEATURE_COLUMNS:
        if base[col].isna().any():
            base[col] = base[col].fillna(base[col].median())
    cyc = cyclical_time_encoding(base["day_of_year"], base["month"])
    base = base.drop(columns=["day_of_year", "month"])
    return pd.concat([base.reset_index(drop=True), cyc.reset_index(drop=True)], axis=1)
