import numpy as np
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    LAND_SHAPEFILE,
    MIN_POSITIVES,
    NON_SPECIES_COLUMNS,
    TRAINING_CSV,
)


def load_dataset():
    df = pd.read_csv(TRAINING_CSV)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df


def tag_ocean_rows(df):
    """Add a boolean `is_ocean` column using the 10m Natural Earth land mask.

    Cells whose centre point falls inside a land polygon are marked False.
    Computes once per unique (lat, lon) pair then merges back.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    land = gpd.read_file(LAND_SHAPEFILE)
    land_union = land.geometry.union_all()

    cells = df[["lat", "lon"]].drop_duplicates().copy()
    cells["is_ocean"] = ~cells.apply(
        lambda r: land_union.contains(Point(r["lon"], r["lat"])), axis=1
    )
    return df.merge(cells, on=["lat", "lon"])


def filter_ocean_only(df):
    """Return only rows whose grid-cell centre is in the ocean."""
    if "is_ocean" not in df.columns:
        df = tag_ocean_rows(df)
    return df[df["is_ocean"]].drop(columns=["is_ocean"]).reset_index(drop=True)


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
