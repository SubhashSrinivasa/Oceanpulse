"""Download iNaturalist research-grade fish observations via the iNaturalist API.

Strategy:
  - Query api.inaturalist.org/v1/observations directly (bypasses GBIF).
  - Filter to research-grade observations of ray-finned fish (taxon_id=47178)
    and elasmobranchs (taxon_id=47346) within the California Current bbox.
  - Paginate 200 records at a time, month by month, to stay within iNaturalist's
    10 000-record-per-query hard limit.
  - All observations are presence = 1 (no confirmed absences).

Results cached as parquet under data/cache/.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    YEARS,
)
from .utils import get_logger, snap_to_grid, standardize_species_name

log = get_logger("process_inaturalist")

INAT_CACHE = CACHE_DIR / "inaturalist_species.parquet"

INAT_API = "https://api.inaturalist.org/v1/observations"
INAT_PAGE_SIZE = 200
INAT_MAX_PER_QUERY = 10_000  # iNaturalist hard cap

# iNaturalist internal taxon IDs (not GBIF taxon keys)
TAXON_IDS = {
    "Actinopterygii": 47178,   # ray-finned fishes
    "Elasmobranchii": 47346,   # sharks and rays
}


def _fetch_inat_month(taxon_name: str, taxon_id: int, year: int, month: int) -> list[dict]:
    """Page through iNaturalist results for one taxon × year × month."""
    records: list[dict] = []
    page = 1
    while True:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "nelat": LAT_MAX,
            "nelng": LON_MAX,
            "swlat": LAT_MIN,
            "swlng": LON_MIN,
            "month": month,
            "year": year,
            "per_page": INAT_PAGE_SIZE,
            "page": page,
            "order": "asc",
            "order_by": "observed_on",
        }
        try:
            r = requests.get(INAT_API, params=params, timeout=60)
            if r.status_code == 429:
                log.warning("iNaturalist rate limit — sleeping 60 s")
                time.sleep(60)
                continue
            if r.status_code != 200:
                log.warning("iNaturalist HTTP %s (taxon=%s year=%d month=%d)", r.status_code, taxon_name, year, month)
                break
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("iNaturalist request error: %s", exc)
            break

        results = data.get("results", [])
        for obs in results:
            date = obs.get("observed_on")
            loc = obs.get("location")  # "lat,lon" string
            taxon = obs.get("taxon") or {}
            name = taxon.get("name") or taxon.get("preferred_common_name")
            if not date or not loc or not name:
                continue
            try:
                lat_s, lon_s = loc.split(",")
                t = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
                if pd.isna(t):
                    continue
                records.append({"time": t, "lat": float(lat_s), "lon": float(lon_s), "species": name})
            except Exception:  # noqa: BLE001
                continue

        total = data.get("total_results", 0)
        fetched = (page - 1) * INAT_PAGE_SIZE + len(results)
        if not results or fetched >= min(total, INAT_MAX_PER_QUERY):
            break
        page += 1
        time.sleep(0.5)

    return records


def fetch_inaturalist_species() -> pd.DataFrame:
    """Return research-grade iNaturalist fish observations in the domain."""
    if INAT_CACHE.exists():
        log.info("iNaturalist cache hit: %s", INAT_CACHE.name)
        return pd.read_parquet(INAT_CACHE)

    from concurrent.futures import ThreadPoolExecutor, as_completed as asc

    tasks = [
        (taxon_name, taxon_id, year, month)
        for year in YEARS
        for month in range(1, 13)
        for taxon_name, taxon_id in TAXON_IDS.items()
    ]

    all_records: list[dict] = []
    # max_workers=4 to stay polite with iNaturalist rate limits
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_fetch_inat_month, tn, tid, yr, mo): (tn, yr, mo)
            for tn, tid, yr, mo in tasks
        }
        for future in asc(futures):
            taxon_name, year, month = futures[future]
            try:
                recs = future.result()
                if recs:
                    log.info("iNat %s %d-%02d: %d records", taxon_name, year, month, len(recs))
                all_records.extend(recs)
            except Exception as exc:  # noqa: BLE001
                log.warning("iNat task failed (%s %d-%02d): %s", taxon_name, year, month, exc)

    if not all_records:
        log.error("No iNaturalist records retrieved — returning empty DataFrame")
        empty = pd.DataFrame(columns=["time", "lat", "lon", "species", "presence", "source"])
        empty.to_parquet(INAT_CACHE, index=False)
        return empty

    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        & (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()

    df["species"] = df["species"].map(standardize_species_name)
    df = df[df["species"] != "unknown"].copy()
    df["presence"] = 1
    df["source"] = "inaturalist"

    df = snap_to_grid(df)
    df = df.drop_duplicates(subset=["time", "lat", "lon", "species"]).reset_index(drop=True)

    out = df[["time", "lat", "lon", "species", "presence", "source"]]
    out.to_parquet(INAT_CACHE, index=False)
    log.info(
        "iNaturalist: %d observations, %d unique species saved -> %s",
        len(out), out["species"].nunique(), INAT_CACHE.name,
    )
    return out
