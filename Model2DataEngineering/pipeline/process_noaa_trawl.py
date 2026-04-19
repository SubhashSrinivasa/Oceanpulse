"""Download marine fish observations from OBIS for the California Current region.

Uses the OBIS (Ocean Biodiversity Information System) v3 API to retrieve
ray-finned fish (Actinopterygii, WoRMS AphiaID 10194) and elasmobranch
(Elasmobranchii, AphiaID 10193) occurrence records within the domain.

All records are presence = 1. Results cached as parquet under data/cache/.
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
    TIME_END,
    TIME_START,
    YEARS,
)
from .utils import get_logger, snap_to_grid, standardize_species_name

log = get_logger("process_noaa_trawl")

TRAWL_CACHE = CACHE_DIR / "noaa_trawl_species.parquet"
OBIS_API = "https://api.obis.org/v3/occurrence"
OBIS_PAGE_SIZE = 5_000

# WoRMS AphiaIDs for fish classes
FISH_TAXON_IDS = {
    "Actinopterygii": 10194,
    "Elasmobranchii": 10193,
}


def _bbox_wkt() -> str:
    return (
        f"POLYGON(({LON_MIN} {LAT_MIN},{LON_MAX} {LAT_MIN},"
        f"{LON_MAX} {LAT_MAX},{LON_MIN} {LAT_MAX},{LON_MIN} {LAT_MIN}))"
    )


def _fetch_obis_page(taxon_id: int, year: int, offset: int) -> dict | None:
    params = {
        "taxonid": taxon_id,
        "geometry": _bbox_wkt(),
        "startdate": f"{year}-01-01",
        "enddate": f"{year}-12-31",
        "size": OBIS_PAGE_SIZE,
        "offset": offset,
    }
    try:
        r = requests.get(OBIS_API, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        log.warning(
            "OBIS HTTP %s (taxon=%d year=%d offset=%d): %s",
            r.status_code, taxon_id, year, offset, r.text[:200],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("OBIS request error (taxon=%d year=%d offset=%d): %s", taxon_id, year, offset, exc)
    return None


def _fetch_obis_year_taxon(taxon_name: str, taxon_id: int, year: int) -> list[dict]:
    records: list[dict] = []
    offset = 0
    while True:
        data = _fetch_obis_page(taxon_id, year, offset)
        if data is None:
            break
        results = data.get("results", [])
        for rec in results:
            lat = rec.get("decimalLatitude")
            lon = rec.get("decimalLongitude")
            date = rec.get("eventDate")
            name = rec.get("species") or rec.get("scientificName")
            if lat is None or lon is None or not date or not name:
                continue
            try:
                t = pd.to_datetime(str(date)[:10], format="%Y-%m-%d", errors="coerce")
                if pd.isna(t):
                    continue
                records.append({"time": t, "lat": float(lat), "lon": float(lon), "species": name})
            except Exception:  # noqa: BLE001
                continue
        total = data.get("total", 0)
        offset += OBIS_PAGE_SIZE
        if not results or offset >= total:
            break
        time.sleep(0.2)

    log.info("OBIS %s year=%d: %d records", taxon_name, year, len(records))
    return records


def fetch_noaa_trawl_species() -> pd.DataFrame:
    """Return marine fish observations from OBIS for the domain."""
    if TRAWL_CACHE.exists():
        log.info("NOAA trawl (OBIS) cache hit: %s", TRAWL_CACHE.name)
        return pd.read_parquet(TRAWL_CACHE)

    from concurrent.futures import ThreadPoolExecutor, as_completed as asc

    tasks = [
        (taxon_name, taxon_id, year)
        for year in YEARS
        for taxon_name, taxon_id in FISH_TAXON_IDS.items()
    ]

    all_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 6)) as executor:
        futures = {
            executor.submit(_fetch_obis_year_taxon, tn, tid, yr): (tn, yr)
            for tn, tid, yr in tasks
        }
        for future in asc(futures):
            taxon_name, year = futures[future]
            try:
                recs = future.result()
                all_records.extend(recs)
            except Exception as exc:  # noqa: BLE001
                log.warning("OBIS task failed (%s %d): %s", taxon_name, year, exc)

    if not all_records:
        log.error("No OBIS records retrieved — returning empty DataFrame")
        empty = pd.DataFrame(columns=["time", "lat", "lon", "species", "presence", "source"])
        empty.to_parquet(TRAWL_CACHE, index=False)
        return empty

    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        & (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()

    t_start = pd.Timestamp(TIME_START)
    t_end = pd.Timestamp(TIME_END)
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)].copy()

    df["species"] = df["species"].map(standardize_species_name)
    df = df[df["species"] != "unknown"].copy()
    df["presence"] = 1
    df["source"] = "obis"

    df = snap_to_grid(df)
    df = df.drop_duplicates(subset=["time", "lat", "lon", "species"]).reset_index(drop=True)

    out = df[["time", "lat", "lon", "species", "presence", "source"]]
    out.to_parquet(TRAWL_CACHE, index=False)
    log.info(
        "OBIS: %d observations, %d unique species -> %s",
        len(out), out["species"].nunique(), TRAWL_CACHE.name,
    )
    return out
